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

    def test_empty_string_returns_unnamed(self):
        # Reaches the `or "unnamed"` fallback: no regex substitutions
        # fire, .lstrip(".") returns "". The fallback exists so an
        # empty --model doesn't produce an empty output filename (which
        # would crash the JSON writer with a directory-vs-file error).
        self.assertEqual(rb.safe_filename_segment(""), "unnamed")

    def test_control_characters_neutralised(self):
        # A newline, tab, or NUL in --model arriving from a shell heredoc
        # or an automated feeder must not reach the output filename or a
        # log line — some filesystems, log pipelines, and terminal
        # sessions do surprising things with these bytes (log injection
        # via embedded CR/LF, ANSI-escape smuggling, filesystem quirks).
        # The property we assert: after sanitisation, every character is
        # printable ASCII in [A-Za-z0-9._-], nothing else.
        for evil in [
            "a\x00b",       # NUL byte
            "a\nb",         # newline
            "a\rb",         # CR
            "a\tb",         # tab
            "a\x1bb",       # ESC (ANSI-escape smuggling)
            "a\x7fb",       # DEL
            "\x00\x00\x00", # all-NUL
        ]:
            with self.subTest(evil=evil):
                out = rb.safe_filename_segment(evil)
                for c in out:
                    self.assertTrue(
                        c.isascii(),
                        f"non-ASCII char {c!r} (U+{ord(c):04X}) in sanitised output {out!r} from input {evil!r}",
                    )
                    self.assertGreaterEqual(
                        ord(c), 0x20,
                        f"control char U+{ord(c):04X} in sanitised output {out!r} from input {evil!r}",
                    )
                    self.assertIn(
                        c, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-",
                        f"disallowed char {c!r} in sanitised output {out!r} from input {evil!r}",
                    )

    def test_unicode_replaced_not_preserved(self):
        # Python 3's re.sub on the ASCII-only class [A-Za-z0-9._-]
        # replaces non-ASCII codepoints (they don't match the safe set).
        # Result: no codepoint above U+007F survives. Model names shared
        # across ecosystems sometimes carry accented characters or CJK
        # names — those should map to dashes, not travel through as-is.
        for unicode_input in ["model-é", "中文", "мoдель", "🚀-fast"]:
            with self.subTest(unicode_input=unicode_input):
                out = rb.safe_filename_segment(unicode_input)
                for c in out:
                    self.assertLessEqual(
                        ord(c), 0x7F,
                        f"non-ASCII codepoint U+{ord(c):04X} survived in {out!r}",
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
        # completion_tokens=10 with a non-zero elapsed interval must
        # yield a positive tpot_ms — the arithmetic at run_benchmark.py:
        # `(end - first_byte) * 1000 / completion_tokens`. A zero tpot_ms
        # in the positive path would mean either the division was
        # skipped or the timer measured negative time, both of which
        # need to be caught here rather than in the aggregated
        # percentile output where the value is folded into a p50.
        self.assertGreater(
            result.tpot_ms, 0,
            f"tpot_ms must be positive when completion_tokens>0 and the response was successful; got {result.tpot_ms}",
        )

    def test_malformed_json_produces_failure_not_silent_success(self):
        # The `except (URLError, TimeoutError, json.JSONDecodeError,
        # ValueError)` clause in make_request has a JSONDecodeError
        # branch that never fires against a real endpoint returning
        # good JSON. But it's the error path for the response-cap fix:
        # a truncated body from a rogue endpoint is exactly what
        # produces malformed JSON, and the cap + this branch have to
        # agree — a cap that silently passes junk through would be
        # worse than no cap. Pin that a body of garbage bytes flows
        # into success=False with the parser error surfaced in `error`.
        garbage = b"<html>Bad Gateway</html>this is not json"

        class FakeResp:
            def __init__(self):
                self._pos = 0

            def read(self, n=None):
                if n is None:
                    n = len(garbage)
                out = garbage[self._pos: self._pos + n]
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

        self.assertFalse(
            result.success,
            f"non-JSON body must yield success=False, got success=True",
        )
        # The error string must be informative — Python's JSONDecodeError
        # str() is of the shape "Expecting value: line 1 column 1 (char
        # 0)". Assert one of the diagnostic keywords so a future
        # refactor that catches and reformats the error can't silently
        # replace it with something opaque like "request failed".
        err_lower = result.error.lower()
        self.assertTrue(
            any(k in err_lower for k in ("json", "expecting", "decode")),
            f"malformed-JSON error must name the failure class; got: {result.error!r}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
