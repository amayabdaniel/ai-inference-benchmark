# AI Inference Benchmark

Rigorous, reproducible benchmark for AI inference in production.

**The question nobody has answered with real data:**
What does it actually cost to serve an LLM per token, across GPU types, inference engines, quantization methods, and multi-model scenarios?

## Why this exists

Every team choosing an inference stack googles for benchmark data and finds:
- Vendor marketing with cherry-picked numbers
- Blog posts testing one model on one GPU
- No cost data, just speed
- No multi-tenant or multi-model scenarios
- Not reproducible

This benchmark is:
- **Reproducible** — Terraform spins up each GPU, runs the benchmark, tears down
- **Comprehensive** — same model across multiple GPUs, engines, and quantization levels
- **Cost-aware** — measures cost per token, not just tokens per second
- **Production-realistic** — includes cold start, concurrent users, multi-model, and sustained load
- **Open data** — raw results in CSV/JSON, analysis notebooks, all code public

## What we measure

### Per-configuration metrics

| Metric | Unit | Why it matters |
|---|---|---|
| Tokens per second (generation) | tok/s | Raw speed |
| Time to first token (TTFT) | ms | User-perceived latency |
| Time per output token (TPOT) | ms | Streaming smoothness |
| End-to-end latency (p50/p95/p99) | ms | SLO planning |
| GPU memory usage | GB | Determines which GPU fits |
| GPU utilization | % | Are you wasting capacity? |
| Cost per 1K tokens | $ | The number that matters for business |
| Tokens per dollar | tok/$ | Efficiency for comparison |
| Cold start time | s | Model loading latency |
| Max concurrent users before degradation | count | Scaling limits |

### Test matrix

**Models:**
- Qwen3 8B (small, fast)
- Llama 3.3 8B (popular)
- Mistral 7B (efficient)
- Qwen3 32B (medium)
- Llama 3.3 70B (large)

**GPU types:**
- T4 (16GB, $0.35/hr) — budget
- L4 (24GB, $0.80/hr) — mid-range
- A10G (24GB, $1.01/hr) — AWS default
- A100-40GB ($3.40/hr) — high-end
- A100-80GB ($4.10/hr) — premium
- H100 ($8.00/hr) — top tier

**Inference engines:**
- vLLM (PagedAttention, dominant OSS)
- Ollama (local-first, developer favorite)
- SGLang (fast, structured generation)

**Quantization:**
- FP16 (baseline)
- Q8 (8-bit)
- Q4_K_M (4-bit, GGUF)
- AWQ (4-bit, vLLM native)

## Methodology

### Workload profiles

Each benchmark runs 3 workload profiles to simulate real usage:

**1. Single-user latency test**
- 100 sequential requests
- 256 prompt tokens, 128 completion tokens
- Measures: TTFT, TPOT, e2e latency percentiles

**2. Throughput test**
- 50 concurrent users, 500 total requests
- Mixed prompt sizes (64-1024 tokens)
- Measures: tokens/sec, GPU utilization, max throughput

**3. Sustained load test**
- 10 concurrent users for 10 minutes
- Realistic prompt distribution
- Measures: latency stability, memory growth, degradation over time

**4. Cold start test**
- Time from `docker run` to first successful inference
- 5 repetitions per configuration
- Measures: model load time, first TTFT

**5. Multi-model test** (when GPU allows)
- 2 models sharing one GPU
- Alternating requests between models
- Measures: per-model degradation, cross-model interference

### Prompt dataset

We use a fixed prompt dataset for reproducibility:
- 100 prompts from ShareGPT (real user conversations)
- Bucketed by length: short (64 tok), medium (256 tok), long (1024 tok)
- Same prompts across all configurations

### Statistical rigor

- Each configuration runs 3 times (cold start runs 5 times)
- Report mean, p50, p95, p99, and standard deviation
- Discard first 10 requests as warmup
- Monitor for thermal throttling (GPU temp >80C invalidates run)

## Running the benchmark

### Prerequisites

- GCP account with GPU quota (or use gpu-lab Terraform)
- Docker + NVIDIA Container Toolkit on the target machine
- Python 3.10+ (for `list[T]` PEP 585 generics used in `run_benchmark.py`)

### Dependencies

`scripts/run_benchmark.py` is **stdlib-only by design** — it uses only
`argparse`, `json`, `re`, `statistics`, `time`, `concurrent.futures`,
`dataclasses`, `datetime`, `pathlib`, `typing`, `urllib`. There is no
`requirements.txt`, `pyproject.toml`, or `setup.py`, and this is
deliberate: a benchmark script that has to bootstrap its own Python
environment before it can measure anything is a benchmark script that
measures the wrong thing. Any change that adds a third-party import
should also state why the stdlib version wasn't enough, and add the
matching pin.

### Quick start

```bash
# Spin up a GPU instance
cd ../gpu-lab && make model-bench-up

# SSH in and run
make model-bench-ssh

# On the instance:
git clone https://github.com/amayabdaniel/ai-inference-benchmark.git
cd ai-inference-benchmark

# Run a single benchmark (e.g., Qwen3 8B on vLLM)
python3 scripts/run_benchmark.py \
  --model qwen3:8b \
  --engine vllm \
  --workload all \
  --output results/
```

Matrix drivers (`run_matrix.py`, `analyze.py`) referenced in earlier
drafts are not yet in the repo — for a full sweep, invoke
`run_benchmark.py` per configuration and aggregate the JSON files from
`results/`. When a matrix driver is added, it will also be stdlib-only.

### Running tests

The security-critical helpers in `run_benchmark.py` are covered by
`scripts/test_run_benchmark.py` (11 cases, stdlib `unittest`):

```bash
python3 -m unittest scripts.test_run_benchmark -v
```

The suite pins the filename-sanitiser containment property (arbitrary
`--model` / `--engine` / `--gpu-type` values cannot escape `--output`)
and the response-body cap (a rogue endpoint returning >8 MiB is
refused, not silently truncated).

### Tear down

```bash
cd ../gpu-lab && make model-bench-down
```

## Results

Results are published in `results/` as CSV and JSON.
Analysis with charts is in `analysis/`.

<!-- Results will be added after benchmark runs -->

## Related projects

- [inferctl](https://github.com/amayabdaniel/inferctl) — the `simulate` command predictions are validated against this benchmark
- [gpucast](https://github.com/amayabdaniel/gpucast) — cost tracking based on the same metrics this benchmark measures
- [gpu-lab](https://github.com/amayabdaniel/gpu-lab) — Terraform infrastructure to run these benchmarks

## License

Apache 2.0
