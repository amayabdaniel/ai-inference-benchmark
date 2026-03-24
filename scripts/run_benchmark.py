#!/usr/bin/env python3
"""
AI Inference Benchmark Runner

Runs standardized benchmarks against any OpenAI-compatible inference endpoint.
Measures latency, throughput, cost, and GPU utilization.
"""

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError


@dataclass
class RequestResult:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ttft_ms: float = 0  # time to first token
    total_ms: float = 0  # end-to-end latency
    tpot_ms: float = 0  # time per output token
    success: bool = True
    error: str = ""


@dataclass
class BenchmarkResult:
    timestamp: str = ""
    model: str = ""
    engine: str = ""
    gpu_type: str = ""
    gpu_hourly_rate: float = 0
    workload: str = ""
    quantization: str = ""
    num_requests: int = 0
    concurrency: int = 0
    duration_seconds: float = 0

    # Latency
    ttft_p50_ms: float = 0
    ttft_p95_ms: float = 0
    ttft_p99_ms: float = 0
    e2e_p50_ms: float = 0
    e2e_p95_ms: float = 0
    e2e_p99_ms: float = 0
    tpot_p50_ms: float = 0

    # Throughput
    tokens_per_second: float = 0
    requests_per_second: float = 0

    # Cost
    cost_per_1k_tokens: float = 0
    tokens_per_dollar: float = 0

    # Errors
    error_count: int = 0
    error_rate: float = 0

    # Raw data
    individual_results: list = field(default_factory=list)


# Default prompts for benchmarking (varying lengths)
PROMPTS_SHORT = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "What is 42 * 37?",
    "Name three programming languages.",
    "What color is the sky?",
]

PROMPTS_MEDIUM = [
    "Explain the difference between TCP and UDP protocols. Include use cases for each.",
    "Write a Python function that implements binary search on a sorted list. Include comments.",
    "Describe the architecture of a typical Kubernetes cluster. What are the main components?",
    "Explain how HTTPS works, including the TLS handshake process step by step.",
    "What are the SOLID principles in software engineering? Give a brief example of each.",
]

PROMPTS_LONG = [
    "Write a detailed technical design document for a real-time chat application that supports "
    "10,000 concurrent users. Cover the architecture, database design, message queuing, "
    "WebSocket management, authentication, and deployment strategy. Include specific technology "
    "choices and justify each decision. Address scalability, fault tolerance, and monitoring.",
] * 5


def make_request(endpoint: str, model: str, prompt: str, max_tokens: int = 128) -> RequestResult:
    """Send a single inference request and measure timing."""
    result = RequestResult()

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }).encode()

    req = Request(
        f"{endpoint}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    try:
        with urlopen(req, timeout=120) as resp:
            first_byte = time.perf_counter()
            body = resp.read()
            end = time.perf_counter()

            data = json.loads(body)
            usage = data.get("usage", {})
            result.prompt_tokens = usage.get("prompt_tokens", 0)
            result.completion_tokens = usage.get("completion_tokens", 0)
            result.total_tokens = usage.get("total_tokens", 0)
            result.ttft_ms = (first_byte - start) * 1000
            result.total_ms = (end - start) * 1000

            if result.completion_tokens > 0:
                result.tpot_ms = (end - first_byte) * 1000 / result.completion_tokens

            result.success = True

    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        end = time.perf_counter()
        result.success = False
        result.error = str(e)
        result.total_ms = (end - start) * 1000

    return result


def percentile(data: list, p: float) -> float:
    """Calculate percentile from a sorted list."""
    if not data:
        return 0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def run_latency_test(endpoint: str, model: str, num_requests: int = 100) -> BenchmarkResult:
    """Single-user sequential latency test."""
    print(f"  Running latency test ({num_requests} sequential requests)...")
    results = []
    prompts = (PROMPTS_SHORT + PROMPTS_MEDIUM) * (num_requests // 10 + 1)

    # Warmup
    for i in range(min(5, num_requests)):
        make_request(endpoint, model, prompts[i])

    for i in range(num_requests):
        r = make_request(endpoint, model, prompts[i % len(prompts)])
        results.append(r)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{num_requests} complete")

    return aggregate_results(results, "latency", 1)


def run_throughput_test(endpoint: str, model: str, concurrency: int = 50,
                        total_requests: int = 500) -> BenchmarkResult:
    """Concurrent throughput test."""
    print(f"  Running throughput test ({total_requests} requests, {concurrency} concurrent)...")
    results = []
    prompts = (PROMPTS_SHORT + PROMPTS_MEDIUM + PROMPTS_LONG) * (total_requests // 15 + 1)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(total_requests):
            f = executor.submit(make_request, endpoint, model, prompts[i % len(prompts)])
            futures.append(f)

        for i, future in enumerate(as_completed(futures)):
            results.append(future.result())
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{total_requests} complete")

    duration = time.perf_counter() - start
    bench = aggregate_results(results, "throughput", concurrency)
    bench.duration_seconds = round(duration, 2)
    return bench


def run_sustained_test(endpoint: str, model: str, concurrency: int = 10,
                       duration_minutes: float = 2) -> BenchmarkResult:
    """Sustained load over time — checks for degradation."""
    duration_sec = duration_minutes * 60
    print(f"  Running sustained test ({concurrency} concurrent, {duration_minutes} min)...")
    results = []
    prompts = (PROMPTS_SHORT + PROMPTS_MEDIUM) * 100

    start = time.perf_counter()
    request_idx = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = set()
        while time.perf_counter() - start < duration_sec:
            # Keep concurrency futures active
            while len(futures) < concurrency:
                f = executor.submit(
                    make_request, endpoint, model,
                    prompts[request_idx % len(prompts)]
                )
                futures.add(f)
                request_idx += 1

            # Collect completed
            done = {f for f in futures if f.done()}
            for f in done:
                results.append(f.result())
                futures.remove(f)

            time.sleep(0.01)

        # Drain remaining
        for f in as_completed(futures):
            results.append(f.result())

    duration = time.perf_counter() - start
    bench = aggregate_results(results, "sustained", concurrency)
    bench.duration_seconds = round(duration, 2)
    print(f"    Completed {len(results)} requests in {duration:.0f}s")
    return bench


def aggregate_results(results: list[RequestResult], workload: str,
                      concurrency: int) -> BenchmarkResult:
    """Aggregate individual results into a benchmark summary."""
    successful = [r for r in results if r.success]
    errors = [r for r in results if not r.success]

    if not successful:
        return BenchmarkResult(workload=workload, error_count=len(errors), error_rate=1.0)

    ttfts = [r.ttft_ms for r in successful if r.ttft_ms > 0]
    e2es = [r.total_ms for r in successful]
    tpots = [r.tpot_ms for r in successful if r.tpot_ms > 0]
    total_tokens = sum(r.total_tokens for r in successful)
    total_completion_tokens = sum(r.completion_tokens for r in successful)
    total_time_sec = sum(r.total_ms for r in successful) / 1000

    bench = BenchmarkResult(
        workload=workload,
        num_requests=len(results),
        concurrency=concurrency,
        error_count=len(errors),
        error_rate=round(len(errors) / len(results), 4) if results else 0,
    )

    if ttfts:
        bench.ttft_p50_ms = round(percentile(ttfts, 50), 1)
        bench.ttft_p95_ms = round(percentile(ttfts, 95), 1)
        bench.ttft_p99_ms = round(percentile(ttfts, 99), 1)

    if e2es:
        bench.e2e_p50_ms = round(percentile(e2es, 50), 1)
        bench.e2e_p95_ms = round(percentile(e2es, 95), 1)
        bench.e2e_p99_ms = round(percentile(e2es, 99), 1)

    if tpots:
        bench.tpot_p50_ms = round(percentile(tpots, 50), 1)

    if total_time_sec > 0:
        bench.tokens_per_second = round(total_completion_tokens / total_time_sec, 1)
        bench.requests_per_second = round(len(successful) / total_time_sec, 2)

    return bench


def add_cost_metrics(bench: BenchmarkResult, gpu_hourly_rate: float):
    """Calculate cost metrics."""
    bench.gpu_hourly_rate = gpu_hourly_rate
    if bench.tokens_per_second > 0 and gpu_hourly_rate > 0:
        tokens_per_hour = bench.tokens_per_second * 3600
        bench.cost_per_1k_tokens = round((gpu_hourly_rate / tokens_per_hour) * 1000, 6)
        bench.tokens_per_dollar = round(tokens_per_hour / gpu_hourly_rate)


def run_benchmark(args):
    """Run the full benchmark for a single configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {args.model} on {args.engine}")
    print(f"Endpoint:  {args.endpoint}")
    print(f"GPU:       {args.gpu_type} (${args.gpu_rate}/hr)")
    print(f"{'='*60}\n")

    results = []

    workloads = args.workload.split(",") if args.workload != "all" else ["latency", "throughput", "sustained"]

    for workload in workloads:
        if workload == "latency":
            bench = run_latency_test(args.endpoint, args.model, num_requests=args.num_requests)
        elif workload == "throughput":
            bench = run_throughput_test(args.endpoint, args.model,
                                        concurrency=args.concurrency,
                                        total_requests=args.num_requests)
        elif workload == "sustained":
            bench = run_sustained_test(args.endpoint, args.model,
                                       concurrency=min(args.concurrency, 10),
                                       duration_minutes=args.duration_minutes)
        else:
            print(f"  Unknown workload: {workload}")
            continue

        bench.timestamp = datetime.utcnow().isoformat()
        bench.model = args.model
        bench.engine = args.engine
        bench.gpu_type = args.gpu_type
        bench.quantization = args.quantization
        add_cost_metrics(bench, args.gpu_rate)

        results.append(bench)
        print_result(bench)

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{args.model.replace(':', '-')}_{args.engine}_{args.gpu_type}.json"
        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def print_result(bench: BenchmarkResult):
    """Print a formatted benchmark result."""
    print(f"\n  --- {bench.workload} results ---")
    print(f"  Requests:     {bench.num_requests} ({bench.error_count} errors, {bench.error_rate*100:.1f}% error rate)")
    if bench.ttft_p50_ms:
        print(f"  TTFT:         p50={bench.ttft_p50_ms}ms  p95={bench.ttft_p95_ms}ms  p99={bench.ttft_p99_ms}ms")
    if bench.e2e_p50_ms:
        print(f"  E2E latency:  p50={bench.e2e_p50_ms}ms  p95={bench.e2e_p95_ms}ms  p99={bench.e2e_p99_ms}ms")
    if bench.tpot_p50_ms:
        print(f"  TPOT:         p50={bench.tpot_p50_ms}ms")
    if bench.tokens_per_second:
        print(f"  Throughput:   {bench.tokens_per_second} tok/s  ({bench.requests_per_second} req/s)")
    if bench.cost_per_1k_tokens:
        print(f"  Cost:         ${bench.cost_per_1k_tokens:.4f}/1K tokens  ({bench.tokens_per_dollar:,.0f} tok/$)")
    print()


def main():
    parser = argparse.ArgumentParser(description="AI Inference Benchmark")
    parser.add_argument("--endpoint", default="http://localhost:8000",
                        help="Inference API endpoint")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--engine", default="vllm", help="Inference engine name (for labeling)")
    parser.add_argument("--gpu-type", default="unknown", help="GPU type (for labeling)")
    parser.add_argument("--gpu-rate", type=float, default=0.80, help="GPU hourly rate in USD")
    parser.add_argument("--quantization", default="fp16", help="Quantization method")
    parser.add_argument("--workload", default="all",
                        help="Workload type: latency, throughput, sustained, or all")
    parser.add_argument("--num-requests", type=int, default=100,
                        help="Number of requests per test")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Concurrent users for throughput/sustained tests")
    parser.add_argument("--duration-minutes", type=float, default=2,
                        help="Duration for sustained test in minutes")
    parser.add_argument("--output", default="results/",
                        help="Output directory for results")

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
