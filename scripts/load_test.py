#!/usr/bin/env python3
"""
Load testing script for vLLM endpoint measuring TTFT, tokens/sec, and end-to-end latency.

Usage:
    python load_test.py --num-requests 10 --concurrency 1
"""

import argparse
import asyncio
import time
from typing import List, Dict, Optional
import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
import pandas as pd
import os


class LoadTester:
    def __init__(self, base_url: str = "http://localhost:8000", input_tokens: int = 2048, output_tokens: int = 1024):
        self.base_url = base_url
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        
    def generate_prompt(self, num_tokens: int) -> str:
        """Generate a prompt of approximately num_tokens length."""
        # Average ~4 characters per token
        text = "The quick brown fox jumps over the lazy dog. " * (num_tokens // 10)
        return text
    
    async def send_request(self, session: aiohttp.ClientSession, request_id: int) -> Dict:
        """Send a single request and measure metrics."""
        prompt = self.generate_prompt(self.input_tokens)
        
        payload = {
            "prompt": prompt,
            "max_tokens": self.output_tokens,
            "temperature": 0.0,
            "stream": True  # Enable streaming to measure TTFT
        }
        
        start_time = time.perf_counter()
        ttft = None
        tokens_received = 0
        
        try:
            async with session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    return {
                        "request_id": request_id,
                        "success": False,
                        "error": f"HTTP {response.status}"
                    }
                
                async for line in response.content:
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    
                    # Count tokens (each SSE event typically contains one token)
                    if line.strip().startswith(b"data:"):
                        tokens_received += 1
        
        except Exception as e:
            return {
                "request_id": request_id,
                "success": False,
                "error": str(e)
            }
        
        end_time = time.perf_counter()
        e2e_latency = end_time - start_time
        
        # Calculate tokens per second (excluding TTFT for more accurate throughput)
        if tokens_received > 0 and e2e_latency > ttft:
            tokens_per_sec = tokens_received / (e2e_latency - ttft)
        else:
            tokens_per_sec = 0
        
        return {
            "request_id": request_id,
            "success": True,
            "ttft_ms": ttft * 1000,
            "e2e_latency_ms": e2e_latency * 1000,
            "tokens_received": tokens_received,
            "tokens_per_sec": tokens_per_sec
        }
    
    async def run_concurrent_requests(self, num_requests: int, concurrency: int) -> List[Dict]:
        """Run multiple requests with specified concurrency."""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(session, req_id):
            async with semaphore:
                return await self.send_request(session, req_id)
        
        async with aiohttp.ClientSession() as session:
            tasks = [bounded_request(session, i) for i in range(num_requests)]
            results = []
            
            # Use tqdm for progress bar
            for coro in tqdm.as_completed(tasks, total=num_requests, desc="Load testing"):
                result = await coro
                results.append(result)
            
            return results
    
    def print_results(self, results: List[Dict]):
        """Print aggregated results."""
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        
        print("\n" + "="*60)
        print("LOAD TEST RESULTS")
        print("="*60)
        
        print(f"\nTotal Requests: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if failed:
            print(f"\nFailure reasons:")
            for f in failed[:5]:  # Show first 5 failures
                print(f"  - Request {f['request_id']}: {f.get('error', 'Unknown')}")
        
        if successful:
            ttfts = [r["ttft_ms"] for r in successful]
            e2e_latencies = [r["e2e_latency_ms"] for r in successful]
            tokens_per_sec = [r["tokens_per_sec"] for r in successful if r["tokens_per_sec"] > 0]
            
            print("\n" + "-"*60)
            print("TTFT (Time To First Token)")
            print("-"*60)
            print(f"  Mean: {np.mean(ttfts):.2f} ms")
            print(f"  Median: {np.median(ttfts):.2f} ms")
            print(f"  P50: {np.percentile(ttfts, 50):.2f} ms")
            print(f"  P90: {np.percentile(ttfts, 90):.2f} ms")
            print(f"  P95: {np.percentile(ttfts, 95):.2f} ms")
            print(f"  P99: {np.percentile(ttfts, 99):.2f} ms")
            print(f"  Min: {np.min(ttfts):.2f} ms")
            print(f"  Max: {np.max(ttfts):.2f} ms")
            
            print("\n" + "-"*60)
            print("End-to-End Latency")
            print("-"*60)
            print(f"  Mean: {np.mean(e2e_latencies):.2f} ms")
            print(f"  Median: {np.median(e2e_latencies):.2f} ms")
            print(f"  P50: {np.percentile(e2e_latencies, 50):.2f} ms")
            print(f"  P90: {np.percentile(e2e_latencies, 90):.2f} ms")
            print(f"  P95: {np.percentile(e2e_latencies, 95):.2f} ms")
            print(f"  P99: {np.percentile(e2e_latencies, 99):.2f} ms")
            print(f"  Min: {np.min(e2e_latencies):.2f} ms")
            print(f"  Max: {np.max(e2e_latencies):.2f} ms")
            
            if tokens_per_sec:
                print("\n" + "-"*60)
                print("Tokens Per Second (Throughput)")
                print("-"*60)
                print(f"  Mean: {np.mean(tokens_per_sec):.2f} tokens/s")
                print(f"  Median: {np.median(tokens_per_sec):.2f} tokens/s")
                print(f"  P50: {np.percentile(tokens_per_sec, 50):.2f} tokens/s")
                print(f"  P90: {np.percentile(tokens_per_sec, 90):.2f} tokens/s")
                print(f"  Min: {np.min(tokens_per_sec):.2f} tokens/s")
                print(f"  Max: {np.max(tokens_per_sec):.2f} tokens/s")
        
        print("\n" + "="*60)
    
    def save_results_to_csv(self, results: List[Dict], csv_path: str, url: str, model_identifier: str, num_requests: int, concurrency: int, input_tokens: int, output_tokens: int):
        """Save P50 metrics to CSV file, appending if file exists."""
        successful = [r for r in results if r.get("success", False)]
        
        if not successful:
            print(f"\nWarning: No successful requests to save to CSV.")
            return
        
        ttfts = [r["ttft_ms"] for r in successful]
        e2e_latencies = [r["e2e_latency_ms"] for r in successful]
        tokens_per_sec = [r["tokens_per_sec"] for r in successful if r["tokens_per_sec"] > 0]
        
        # Calculate P50 values
        ttft_p50 = np.percentile(ttfts, 50)
        e2e_latency_p50 = np.percentile(e2e_latencies, 50)
        tokens_per_sec_p50 = np.percentile(tokens_per_sec, 50) if tokens_per_sec else 0.0
        
        # Create summary row
        summary_row = {
            "model": model_identifier,
            "url": url,
            "num_requests": num_requests,
            "concurrency": concurrency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "successful_requests": len(successful),
            "failed_requests": len(results) - len(successful),
            "ttft_p50_ms": round(ttft_p50, 2),
            "tokens_per_sec_p50": round(tokens_per_sec_p50, 2),
            "e2e_latency_p50_ms": round(e2e_latency_p50, 2)
        }
        
        # Append to existing CSV or create new one
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        else:
            df = pd.DataFrame([summary_row])
        
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")


async def main():
    parser = argparse.ArgumentParser(description="Load test vLLM endpoint")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the vLLM server")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model identifier (e.g., 'baseline', 'optimized_v1')")
    parser.add_argument("--num-requests", "-n", type=int, default=10, help="Number of requests to send")
    parser.add_argument("--concurrency", "-c", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--input-tokens", type=int, default=2048, help="Approximate input tokens")
    parser.add_argument("--output-tokens", type=int, default=1024, help="Max output tokens")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file to save/append results (P50 metrics)")
    
    args = parser.parse_args()
    
    print(f"Starting load test:")
    print(f"  Model: {args.model}")
    print(f"  URL: {args.url}")
    print(f"  Num Requests: {args.num_requests}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Input Tokens: {args.input_tokens}")
    print(f"  Output Tokens: {args.output_tokens}")
    print()
    
    tester = LoadTester(
        base_url=args.url,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens
    )
    
    results = await tester.run_concurrent_requests(args.num_requests, args.concurrency)
    tester.print_results(results)
    
    # Save to CSV if path provided
    if args.csv:
        tester.save_results_to_csv(
            results, 
            args.csv, 
            args.url,
            args.model,
            args.num_requests, 
            args.concurrency,
            args.input_tokens,
            args.output_tokens
        )


if __name__ == "__main__":
    asyncio.run(main())

