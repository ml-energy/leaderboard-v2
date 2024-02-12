"""Taken and modified from vllm: https://github.com/vllm-project/vllm/blob/93b38bea5dd03e1b140ca997dfaadef86f8f1855/benchmarks/benchmark_serving.py
   Identical to benchmark_server.py, but additionally calculates total energy as reported by Zeus.
"""

import argparse
import asyncio
import json
import random
import time
import torch
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from zeus.monitor import ZeusMonitor

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []
SYSTEM_PROMPT = "A chat between a human user (prompter) and an artificial intelligence (AI) assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
TOTAL_ENERGY = 0
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0


# @dataclass
# class Results:
#     total_time: float
#     total_energy: float
#     time_per_request: float
#     time_per_token: float
#     energy_per_request: float
#     energy_per_token: float
#     results: list[Result]
#     ? server: str ? (tgi vs vllm) TODO

# @dataclass
# class Result:in py
#     prompt: str
#     response: str
#     prompt_len_in_tokens: int
#     response_len_in_tokens: int


def get_requests(
    dataset_path: str,
) -> List[str]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Only keep the first turn of each conversation.
    dataset = [data["conversations"][0]["value"] for data in dataset]

    return dataset


async def get_request(
    input_requests: List[str],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    backend: str,
    model: str,
    api_url: str,
    prompt: str,
    pbar: tqdm,
) -> None:
    request_start_time = time.perf_counter()

    # headers = {"User-Agent": "Benchmark Client"}
    headers = {"Content-Type": "application/json"}
    # Both tgi and vllm support OpenAI Chat Completion API
    if backend in ["tgi" or "vllm"]:
        pload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "max_tokens": 1000,  # tgi: `inputs` tokens + `max_new_tokens` must be <= 2048
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append(
        (
            output["usage"]["prompt_tokens"],
            output["usage"]["completion_tokens"],
            request_latency,
        )
    )

    # track energy usage
    global TOTAL_ENERGY
    global TOTAL_PROMPT_TOKENS
    global TOTAL_COMPLETION_TOKENS
    TOTAL_ENERGY += output["usage"]["energy"]
    TOTAL_PROMPT_TOKENS += output["usage"]["prompt_tokens"]
    TOTAL_COMPLETION_TOKENS += output["usage"]["completion_tokens"]

    pbar.update(1)


async def benchmark(
    backend: str,
    model: str,
    api_url: str,
    input_requests: List[str],
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    pbar = tqdm(total=len(input_requests))
    async for request in get_request(input_requests, request_rate):
        prompt = request
        task = asyncio.create_task(send_request(backend, model, api_url, prompt, pbar))
        tasks.append(task)
    await asyncio.gather(*tasks)
    pbar.close()


def run_benchmark(args: argparse.Namespace, api_url: str, input_requests: List[str], out_filename: str):
    zeus_monitor = ZeusMonitor()
    zeus_monitor.begin_window(out_filename)
    benchmark_start_time = time.perf_counter()
    asyncio.run(
        benchmark(
            args.backend,
            args.model,
            api_url,
            input_requests,
            args.request_rate,
        )
    )
    benchmark_end_time = time.perf_counter()
    measurements = zeus_monitor.end_window(out_filename)
    zeus_total_energy = measurements.total_energy

    with open(out_filename, "w") as f:
        benchmark_time = benchmark_end_time - benchmark_start_time
        f.write(f"Total time: {benchmark_time:.2f} s\n")
        f.write(f"Throughput: {len(input_requests) / benchmark_time:.2f} requests/s\n\n")

        # Compute the latency statistics
        avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
        f.write(f"Average latency: {avg_latency:.2f} s\n")
        avg_per_token_latency = np.mean(
            [
                latency / (prompt_len + output_len)
                for prompt_len, output_len, latency in REQUEST_LATENCY
            ]
        )
        f.write(f"Average latency per token: {avg_per_token_latency:.2f} s\n")
        avg_per_output_token_latency = np.mean(
            [latency / output_len for _, output_len, latency in REQUEST_LATENCY]
        )
        f.write("Average latency per output token: " f"{avg_per_output_token_latency:.2f} s\n\n")

        # Compute the energy statistics
        f.write(f"Total energy (Zeus): {zeus_total_energy:.2f} J\n")
        f.write(f"Total energy (Individual responses): {TOTAL_ENERGY:.2f} J\n")
        f.write(f"Total prompt tokens: {TOTAL_PROMPT_TOKENS}\n")
        f.write(f"Total completion tokens: {TOTAL_COMPLETION_TOKENS}\n\n")

        f.write("Based on Zeus\n")
        f.write(f"Energy per request: {zeus_total_energy / len(input_requests):.2f} J\n")
        energy_per_token = zeus_total_energy / TOTAL_COMPLETION_TOKENS
        f.write(f"Energy per token: {energy_per_token:.2f} J\n\n")

        f.write("Based on individual responses\n")
        f.write(f"Energy per request: {TOTAL_ENERGY / len(input_requests):.2f} J\n")
        energy_per_token = TOTAL_ENERGY / TOTAL_COMPLETION_TOKENS
        f.write(f"Energy per token: {energy_per_token:.2f} J\n\n")

    print("Benchmark results written to", out_filename)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_filename = args.out_filename
    api_url = f"{args.protocol}://{args.host}:{args.port}{args.endpoint}"
    input_requests = get_requests(args.dataset)

    # run multiple times to warm up
    for i in range(args.num_runs):
        run_benchmark(args, api_url, input_requests, out_filename+f"-run{i}.txt")

    # TODO: concurrency bug. Currently accumlulates TOTAL_ENERGY across sequential runs
    # reset global variables
    REQUEST_LATENCY = []
    TOTAL_ENERGY = 0
    TOTAL_PROMPT_TOKENS = 0
    TOTAL_COMPLETION_TOKENS = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "tgi"])
    parser.add_argument(
        "--protocol", type=str, default="http", choices=["http", "https"]
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", type=str, default="/v1/chat/completions")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Runs the benchmark num-runs times, writing results to 3 separate files.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--out-filename",
        type=str,
        default="benchmark_results.txt",
        help="Name of file to write benchmark results.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    args = parser.parse_args()
    main(args)
