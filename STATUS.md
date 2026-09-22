# STATUS

**Last updated:** 2026-09-22
**Recommendation:** Maintain with fixes; ship no new features until one of the trigger conditions below fires.

## What this repo is today

- One stdlib-only Python script (`scripts/run_benchmark.py`, 389 lines)
  that hits any OpenAI-compatible `/v1/chat/completions` endpoint and
  emits a JSON summary with latency percentiles, throughput,
  cost-per-1K-tokens, and per-request raw data.
- One test file (`scripts/test_run_benchmark.py`, 11 cases,
  stdlib `unittest`) covering the two security-critical helpers
  introduced 2026-09-19: `safe_filename_segment` (with a containment
  property test across 7 evil inputs) and the `MAX_RESPONSE_BYTES`
  response-body cap in `make_request` (with a mocked-oversize failure
  case and a small-body positive companion).
- One README describing the intended matrix and workload profiles,
  plus a mock-result JSON in `results/`.
- No `requirements.txt` / `pyproject.toml` / `setup.py` — the
  stdlib-only property is enforced by convention, not by tooling.

## What this repo is NOT

- **Not run against real GPUs yet.** The results table in the README
  ("Qwen3 8B on L4 vs A10G vs H100 …") is aspirational. Actual sweeps
  require the `gpu-lab` component `model-bench` to be up.
- **Not a matrix driver.** `run_matrix.py` and `analyze.py` are
  referenced in prior drafts but do not exist. A future matrix driver
  should also be stdlib-only.
- **Not a service.** It's a one-shot script an operator invokes on a
  GPU instance; there's no long-running process, no scheduler, no
  persistence beyond the JSON files.

## Known-safe properties (as of 2026-09-22)

- `--model` / `--engine` / `--gpu-type` values are sanitised through
  `safe_filename_segment` before being joined to `--output`; a
  belt-and-suspenders parent check ensures the resolved path stays
  inside `--output` even if the sanitiser is ever weakened.
- Response bodies over 8 MiB are refused with `success=False` and an
  explicit error string, rather than silently truncated (which would
  parse as malformed JSON and look like a real 500 from the endpoint).
- `go`-style Python: no third-party dependencies, no
  `pip install` step, no `venv`-management before the benchmark runs.

## When to touch this repo

Do work here only when at least one of these fires:

1. **A real GPU sweep is being run.** Add whatever driver / harness is
   needed to make the sweep reproducible, but keep the constraint:
   stdlib-only, no new deps without a documented reason.
2. **A benchmark result changes an inferctl `simulate` prediction.**
   The `simulate` command in `inferctl` claims heuristics; when a real
   run contradicts one, update the constant in `pkg/models/simulate.go`
   (in the inferctl repo) and add the datapoint to `results/` here.
3. **A security finding lands on `run_benchmark.py`.** The Saturday
   audit (2026-09-19) closed two: filename sanitisation and response
   cap. Any future audit should add its findings to
   `test_run_benchmark.py` before shipping the fix, not after.
4. **A new inference engine ships that changes the OpenAI-compatible
   API contract.** The script assumes `usage.{prompt,completion,total}_tokens`
   in the response. If an engine emits a different schema, either
   patch here or wrap that engine to conform.
5. **The stdlib-only property is broken.** If a future contributor
   adds a third-party import, the fix is either revert or add the
   corresponding pin AND a justification comment in the module docstring.

If none of the above fires, this repo is idle by design. It has been
cold for months at a time and that is fine — a benchmark script's
value is in the property that it hasn't changed, not in commit
cadence.

## Related repos

- `inferctl` — the `simulate` command's heuristics are what this
  benchmark validates.
- `gpu-lab` — the Terraform that spins up the target GPU instance
  (`model-bench` component) this script runs against.
