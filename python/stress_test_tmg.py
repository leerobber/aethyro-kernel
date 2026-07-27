#!/usr/bin/env python3
"""
Stress test: TMG with many intents.

Generates 100-1000 unique intents and measures:
- NanoKeymaster latency under load
- TMG topology growth (intents, edges)
- Similarity search performance at scale
- Memory footprint
"""

import subprocess
import json
import time
import sys
from collections import defaultdict
from pathlib import Path


def generate_code_intents(count: int) -> list[str]:
    """Generate realistic Rust code intent samples."""
    templates = [
        "fn {name}() {{ }}",
        "struct {name} {{ field: i32 }}",
        "impl {name} {{ pub fn new() -> Self {{ }} }}",
        "trait {name} {{ fn execute(&self); }}",
        "use crate::{name};",
        "const {name}: usize = 42;",
        "let {name} = vec![1, 2, 3];",
        "#[derive({name})]",
        "pub enum {name} {{ A, B, C }}",
        "async fn {name}() -> Result<(), Box<dyn std::error::Error>> {{ Ok(()) }}",
        "macro_rules! {name} {{ ($x:expr) => {{ $x * 2 }} }}",
        "unsafe {{ {name}(); }}",
        "#[test] fn test_{name}() {{ assert!(true); }}",
        "impl From<String> for {name} {{ fn from(s: String) -> Self {{ }} }}",
        "match {name} {{ Some(x) => x, None => 0 }}",
    ]

    intents = []
    for i in range(count):
        template = templates[i % len(templates)]
        intent_name = f"intent_{i:04d}"
        code = template.format(name=intent_name)
        intents.append(code)

    return intents


def run_stress_test(intent_count: int, batch_size: int = 10) -> dict:
    """Run NanoKeymaster with many intents and collect metrics."""
    print(f"\n{'='*70}")
    print(f"Stress Test: {intent_count} intents, batch size {batch_size}")
    print(f"{'='*70}\n")

    # Generate intents
    intents = generate_code_intents(intent_count)
    print(f"[*] Generated {len(intents)} unique code intent samples")

    # Prepare JSON requests
    requests = []
    for i, code in enumerate(intents):
        intent_type = ["classify", "score", "predict"][i % 3]
        req = {
            "intent": intent_type,
            "payload": {"text": code}
        }
        requests.append(req)

    print(f"[*] Prepared {len(requests)} JSON requests")

    # Run NanoKeymaster
    print(f"\n[*] Starting NanoKeymaster...")
    process = subprocess.Popen(
        ["cargo", "run", "--release", "--bin", "kernel_host"],
        cwd="/workspace/aethyro-kernel/kernel",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Metrics collection
    metrics = {
        "total_requests": len(requests),
        "total_time_sec": 0,
        "min_latency_us": float('inf'),
        "max_latency_us": 0,
        "avg_latency_us": 0,
        "throughput_req_sec": 0,
        "final_intent_count": 0,
        "final_edge_count": 0,
        "routing_stats": defaultdict(int),
        "request_times": [],
    }

    responses = []
    errors = []

    # Send requests and collect responses
    t_start = time.time()

    try:
        for i, req in enumerate(requests):
            # Send request
            req_json = json.dumps(req) + "\n"
            process.stdin.write(req_json)
            process.stdin.flush()

            # Read response
            resp_line = process.stdout.readline()
            if not resp_line:
                break

            try:
                resp = json.loads(resp_line)
                responses.append(resp)

                # Extract metrics
                if "latency_us" in resp:
                    latency = resp["latency_us"]
                    metrics["request_times"].append(latency)
                    metrics["min_latency_us"] = min(metrics["min_latency_us"], latency)
                    metrics["max_latency_us"] = max(metrics["max_latency_us"], latency)

                if "backend" in resp:
                    metrics["routing_stats"][resp["backend"]] += 1

                if "tmg_stats" in resp:
                    metrics["final_intent_count"] = resp["tmg_stats"]["intents"]
                    metrics["final_edge_count"] = resp["tmg_stats"]["edges"]

                # Progress
                if (i + 1) % batch_size == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    print(
                        f"  [{i+1:4d}/{len(requests)}] {rate:6.1f} req/s | "
                        f"TMG: {metrics['final_intent_count']} intents, {metrics['final_edge_count']} edges"
                    )

            except json.JSONDecodeError as e:
                errors.append(f"Response {i}: {e}")

        # Clean exit
        print("\n[*] Sending status query...")
        process.stdin.write('{"intent":"_status"}\n')
        process.stdin.flush()
        status_line = process.stdout.readline()
        if status_line:
            status = json.loads(status_line)
            print(f"[+] Status: {json.dumps(status, indent=2)}")

        print("\n[*] Shutting down...")
        process.stdin.write('{"intent":"_quit"}\n')
        process.stdin.flush()

    except Exception as e:
        print(f"[!] Error during test: {e}")
        errors.append(str(e))
    finally:
        # Finalize metrics
        elapsed = time.time() - t_start
        metrics["total_time_sec"] = elapsed

        if metrics["request_times"]:
            metrics["avg_latency_us"] = sum(metrics["request_times"]) / len(metrics["request_times"])
            metrics["throughput_req_sec"] = len(metrics["request_times"]) / elapsed

        # Terminate process
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    return metrics, responses, errors


def print_results(intent_count: int, metrics: dict, errors: list):
    """Print formatted test results."""
    print(f"\n{'='*70}")
    print(f"Results: {intent_count} Intents")
    print(f"{'='*70}\n")

    print(f"Latency (microseconds):")
    print(f"  Min:     {metrics['min_latency_us']:8.1f} µs")
    print(f"  Max:     {metrics['max_latency_us']:8.1f} µs")
    print(f"  Average: {metrics['avg_latency_us']:8.1f} µs")

    print(f"\nThroughput:")
    print(f"  Total requests:  {metrics['total_requests']}")
    print(f"  Total time:      {metrics['total_time_sec']:.2f} seconds")
    print(f"  Rate:            {metrics['throughput_req_sec']:6.1f} requests/second")

    print(f"\nTMG Memory (Final State):")
    print(f"  Intents:  {metrics['final_intent_count']}")
    print(f"  Edges:    {metrics['final_edge_count']}")
    if metrics['final_intent_count'] > 0:
        avg_degree = 2 * metrics['final_edge_count'] / max(1, metrics['final_intent_count'])
        print(f"  Avg degree: {avg_degree:.2f} edges/intent")

    print(f"\nRouting Distribution:")
    total_routed = sum(metrics['routing_stats'].values())
    for backend, count in sorted(metrics['routing_stats'].items()):
        pct = (count / total_routed * 100) if total_routed > 0 else 0
        print(f"  {backend:15s}: {count:5d} ({pct:5.1f}%)")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors[:5]:  # Show first 5 errors
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


def main():
    print("\n" + "="*70)
    print("TMG Stress Test Suite")
    print("="*70)

    test_configs = [
        (50, 10),
        (100, 10),
        (200, 20),
        (500, 50),
        (1000, 50),
    ]

    all_results = {}

    for intent_count, batch_size in test_configs:
        try:
            metrics, responses, errors = run_stress_test(intent_count, batch_size)
            print_results(intent_count, metrics, errors)
            all_results[intent_count] = metrics

        except Exception as e:
            print(f"[!] Test failed for {intent_count} intents: {e}")
            all_results[intent_count] = {"error": str(e)}

    # Summary comparison
    print(f"\n{'='*70}")
    print("Scaling Analysis")
    print(f"{'='*70}\n")

    print(f"{'Intents':>8} | {'Latency (µs)':>12} | {'Throughput':>12} | {'TMG Intents':>12} | {'TMG Edges':>10}")
    print("-" * 70)

    for intent_count in sorted(all_results.keys()):
        m = all_results[intent_count]
        if "error" not in m:
            print(
                f"{intent_count:8d} | "
                f"{m['avg_latency_us']:12.1f} | "
                f"{m['throughput_req_sec']:12.1f} | "
                f"{m['final_intent_count']:12d} | "
                f"{m['final_edge_count']:10d}"
            )

    print(f"\n[+] Stress test complete!")


if __name__ == "__main__":
    main()
