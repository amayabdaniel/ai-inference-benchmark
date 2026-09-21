"""Tests for the two security-critical functions introduced 2026-09-19.

Kept in stdlib `unittest` — same "zero-dependency Python script" property
the rest of `run_benchmark.py` holds; adding pytest or requirements just
to run these tests would defeat the point of that constraint.

Run with:

    python3 -m unittest scripts.test_run_benchmark -v

Or from the scripts/ directory:

    python3 -m unittest test_run_benchmark -v
"""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_run_benchmark():
    # Import the sibling module by path so the tests keep working
    # regardless of the current working directory.
    spec = importlib.util.spec_from_file_location(
        "run_benchmark", os.path.join(_HERE, "run_benchmark.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rb = _load_run_benchmark()


class SafeFilenameSegmentTests(unittest.TestCase):
    # The sanitiser replaces anything outside [A-Za-z0-9._-] with `-`,
    # strips leading dots, and returns `unnamed` for degenerate cases.
    # The security property under test is that the RETURNED value is
    # always a single-filename component that cannot escape a `--output`
    # directory when joined with `Path(output_dir) / result`.

    def test_typical_model_name(self):
        # Colons are common in Ollama-style model refs; they must not
        # break the filename.
        self.assertEqual(rb.safe_filename_segment("qwen3:8b"), "qwen3-8b")

    def test_slashes_become_dashes(self):
        # This is the security-critical case. Without the sanitiser
        # (or with the old ":"-only replacement), `foo/bar` produced
        # a two-segment path that `Path(output_dir) / segment` would
        # interpret as a subdirectory.
        self.assertEqual(rb.safe_filename_segment("foo/bar"), "foo-bar")

    def test_traversal_flattened_to_single_segment(self):
        # `../../etc/passwd` becomes something ugly but single-flat.
        # Dots are legitimate filename characters (`.tar.gz`), so we
        # keep them — the load-bearing invariant is "no path
        # separators", not "no dots".
        got = rb.safe_filename_segment("../../etc/passwd")
        self.assertNotIn("/", got)
        self.assertNotIn("\\", got)
        # No leading dots either.
        self.assertFalse(got.startswith("."))

    def test_leading_dot_stripped(self):
        # ".env" would be a hidden file, and lets a caller controlling
        # the model name produce shell-hidden output files.
        self.assertEqual(rb.safe_filename_segment(".env"), "env")

    def test_all_dots_returns_unnamed(self):
        # ".", "..", "..." all resolve to path components that either
        # refer to cwd/parent or start with dots. After stripping
        # leading dots the string is empty, so the sanitiser returns
        # a fixed placeholder instead of an empty filename.
        for evil in [".", "..", "....", "..."]:
            with self.subTest(evil=evil):
                self.assertEqual(rb.safe_filename_segment(evil), "unnamed")

    def test_spaces_and_symbols_become_dashes(self):
        self.assertEqual(rb.safe_filename_segment("has spaces"), "has-spaces")
        self.assertEqual(rb.safe_filename_segment("has#special!"), "has-special-")

    def test_preserves_legitimate_dots_underscores_dashes(self):
        # These characters are what the sanitiser explicitly allows.
        # If a future edit tightens the regex, this test forces the
        # editor to consider the impact on real model names.
        self.assertEqual(
            rb.safe_filename_segment("model_v1.2.3-base"), "model_v1.2.3-base"
        )


class FilenameContainmentTests(unittest.TestCase):
    # The end-to-end property: joining `Path(output_dir) /
    # safe_filename_segment(evil)` MUST stay inside output_dir.

    def test_evil_inputs_stay_under_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            for evil in [
                "../../etc/passwd",
                "/etc/passwd",
                "../../../root",
                "..",
                ".",
                "foo/bar/baz",
                "\\..\\..\\windows\\path",
            ]:
                with self.subTest(evil=evil):
                    seg = rb.safe_filename_segment(evil)
                    joined = (root / seg).resolve()
                    # `joined` must be a direct child of root.
                    self.assertEqual(
                        joined.parent, root,
                        f"input={evil!r} sanitised={seg!r} joined={joined} escaped {root}",
                    )


class ResponseCapTests(unittest.TestCase):
    # The response-body cap prevents a rogue endpoint from making the
    # script allocate an unbounded body into memory. Two load-bearing
    # properties: (1) the cap exists and is a reasonable size, (2)
    # hitting the cap raises rather than silently truncating.

    def test_max_response_bytes_is_set_reasonably(self):
        # 1 MiB lower bound: chat completions with reasoning traces +
        # function-call metadata can approach hundreds of KiB. 100 MiB
        # upper bound: past that we're back to "effectively unbounded".
        self.assertGreaterEqual(rb.MAX_RESPONSE_BYTES, 1 * 1024 * 1024)
        self.assertLessEqual(rb.MAX_RESPONSE_BYTES, 100 * 1024 * 1024)

    def test_make_request_rejects_oversized_body(self):
        # A response body larger than the cap must produce success=False
        # on the RequestResult, not a silent truncation that parses as
        # malformed JSON (which would look like a real 500 from the
        # endpoint and hide the fact that the cap fired).
        oversized = b"x" * (rb.MAX_RESPONSE_BYTES + 1)

        class FakeResp:
            def read(self, n=None):
                if n is None or n > len(oversized):
                    return oversized
                return oversized[:n]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        original = rb.urlopen
        rb.urlopen = lambda *a, **kw: FakeResp()
        try:
            result = rb.make_request("http://fake", "test-model", "hello")
        finally:
            rb.urlopen = original

        self.assertFalse(
            result.success,
            f"oversized response must yield success=False, got {result}",
        )
        self.assertIn(
            "response body exceeded", result.error,
            f"error must name the cap; got: {result.error!r}",
        )

    def test_make_request_accepts_body_under_cap(self):
        # Positive companion: a legitimate small body succeeds. Guards
        # against a regression that mistakes normal-sized responses
        # for oversized ones.
        small_body = b'{"usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}}'

        class FakeResp:
            def __init__(self):
                self._pos = 0

            def read(self, n=None):
                if n is None:
                    n = len(small_body)
                out = small_body[self._pos: self._pos + n]
                self._pos += len(out)
                return out

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        original = rb.urlopen
        rb.urlopen = lambda *a, **kw: FakeResp()
        try:
            result = rb.make_request("http://fake", "test-model", "hello")
        finally:
            rb.urlopen = original

        self.assertTrue(
            result.success,
            f"small body must succeed; got success={result.success} error={result.error!r}",
        )
        self.assertEqual(result.prompt_tokens, 5)
        self.assertEqual(result.completion_tokens, 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
