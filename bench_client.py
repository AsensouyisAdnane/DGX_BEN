"""
bench_client.py
-----------------
Fires requests at an OpenAI-compatible /v1/completions endpoint (vLLM,
TRT-LLM, and NIM all expose this) and measures, per request:
  - TTFT (time to first token) via streaming
  - total latency
  - tokens generated
Then aggregates across a concurrency sweep to get throughput and a
degradation curve, which is the real story for "multiple request handling".
"""

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests


@dataclass
class RequestResult:
    concurrency: int
    success: bool
    ttft_s: float = None
    total_latency_s: float = None
    input_tokens: int = None
    output_tokens: int = None
    error: str = None


def send_one_request(base_url: str, prompt: str, max_tokens: int, timeout_s: int,
                      concurrency: int, model_name: str, temperature: float,
                      top_p: float) -> RequestResult:
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": temperature,
        "top_p": top_p,
        "stream_options": {"include_usage": True},
    }
    start = time.time()
    first_token_time = None
    output_tokens = 0

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                if first_token_time is None:
                    first_token_time = time.time()
                try:
                    chunk = json.loads(data)
                    usage = chunk.get("usage") or {}
                    if usage.get("completion_tokens") is not None:
                        output_tokens = usage["completion_tokens"]
                    if usage.get("prompt_tokens") is not None:
                        input_tokens = usage["prompt_tokens"]
                    # completions API: choices[0]["text"]; count roughly via whitespace
                    text_piece = chunk.get("choices", [{}])[0].get("text", "")
                    if text_piece:
                        output_tokens += max(1, len(text_piece.split()))
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        end = time.time()
        if first_token_time is None:
            # Non-streaming server, or empty response: fall back gracefully
            first_token_time = end
        return RequestResult(
            concurrency=concurrency,
            success=True,
            ttft_s=first_token_time - start,
            total_latency_s=end - start,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as e:
        return RequestResult(
            concurrency=concurrency,
            success=False,
            error=str(e)[:300],
        )


def run_concurrency_level(base_url: str, concurrency: int, num_requests: int,
                           prompt: str, max_tokens: int, timeout_s: int,
                           model_name: str, temperature: float, top_p: float) -> dict:
    results = []
    wall_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(send_one_request, base_url, prompt, max_tokens, timeout_s,
                        concurrency, model_name, temperature, top_p)
            for _ in range(num_requests)
        ]
        for f in as_completed(futures):
            results.append(f.result())
    wall_elapsed = time.time() - wall_start

    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    total_tokens = sum(r.output_tokens or 0 for r in ok)

    def pct(vals, p):
        if not vals:
            return None
        s = sorted(vals)
        idx = min(len(s) - 1, int(len(s) * p))
        return s[idx]

    ttfts = [r.ttft_s for r in ok]
    latencies = [r.total_latency_s for r in ok]

    return {
        "concurrency": concurrency,
        "requests_sent": num_requests,
        "requests_succeeded": len(ok),
        "requests_failed": len(failed),
        "wall_clock_s": wall_elapsed,
        "throughput_tokens_per_s": (total_tokens / wall_elapsed) if wall_elapsed > 0 else None,
        "throughput_req_per_s": (len(ok) / wall_elapsed) if wall_elapsed > 0 else None,
        "ttft_mean_s": statistics.mean(ttfts) if ttfts else None,
        "ttft_p50_s": pct(ttfts, 0.50),
        "ttft_p95_s": pct(ttfts, 0.95),
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "latency_p50_s": pct(latencies, 0.50),
        "latency_p95_s": pct(latencies, 0.95),
        "sample_errors": [r.error for r in failed[:3]],  # first few, for debugging
        "raw_results": results,
    }


def run_full_sweep(base_url: str, concurrency_levels: list, num_requests_per_level: int,
                    prompt: str, max_tokens: int, timeout_s: int, model_name: str,
                    temperature: float, top_p: float) -> dict:
    """Runs every concurrency level and returns both per-level results and a
    single-row summary (used for the main results CSV)."""
    per_level = []
    for c in concurrency_levels:
        level_result = run_concurrency_level(
            base_url, c, num_requests_per_level, prompt, max_tokens, timeout_s,
            model_name, temperature, top_p
        )
        per_level.append(level_result)
        # If EVERY request failed at this concurrency, no point climbing higher
        if level_result["requests_succeeded"] == 0:
            break

    # Overall summary: use the highest concurrency level that had >=95%
    # success as the "sustainable" throughput number for the summary row.
    sustainable = [lvl for lvl in per_level if lvl["requests_sent"] > 0
                   and lvl["requests_succeeded"] / lvl["requests_sent"] >= 0.95]
    best = sustainable[-1] if sustainable else (per_level[0] if per_level else None)

    max_concurrency_reached = max((lvl["concurrency"] for lvl in per_level
                                    if lvl["requests_succeeded"] > 0), default=0)

    return {
        "per_level": per_level,
        "best_level_summary": best,
        "max_stable_concurrency": max_concurrency_reached,
        "overall_success": any(lvl["requests_succeeded"] > 0 for lvl in per_level),
    }
