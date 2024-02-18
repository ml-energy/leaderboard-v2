"""Taken and modified from vllm: https://github.com/vllm-project/vllm/blob/93b38bea5dd03e1b140ca997dfaadef86f8f1855/benchmarks/benchmark_serving.py
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
from dataclasses import asdict, dataclass, field
from tqdm.asyncio import tqdm
from zeus.monitor import ZeusMonitor


SYSTEM_PROMPT = "A chat between a human user (prompter) and an artificial intelligence (AI) assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "


@dataclass
class Results:
    model: str
    backend: str
    request_rate: float
    num_failures: int = 0
    system_prompt: str = SYSTEM_PROMPT
    total_time: float = 0.0
    throughput: float = 0.0
    total_prompt_tokens: int = 0.0
    total_completion_tokens: int = 0.0
    avg_latency: float = 0.0
    avg_latency_per_token: float = 0.0
    avg_latency_per_output_token: float = 0.0
    server_total_energy: float = 0.0
    server_energy_per_request: float = 0.0
    server_energy_per_output_token: float = 0.0
    local_zeus_total_energy: float = 0.0
    local_zeus_energy_per_request: float = 0.0
    local_zeus_energy_per_output_token: float = 0.0
    results: list["Result"] = field(default_factory=list)


@dataclass
class Result:
    success: bool = True
    latency: float = 0.0
    prompt: str = ""
    response: str = ""
    num_prompt_tokens: int = 0
    num_completion_tokens: int = 0
    energy: float = 0.0


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
    for i, request in enumerate(input_requests):
        yield i, request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    result: Result,
    backend: str,
    model: str,
    api_url: str,
    prompt: str,
    pbar: tqdm,
) -> None:
    request_start_time = time.perf_counter()

    headers = {"Content-Type": "application/json"}
    # OpenAI Chat Completions API request format
    pload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": 1000,
    }

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(api_url, headers=headers, json=pload) as response:
            # Request failed
            if response.status // 100 != 2:
                print('request failed')
                print(f"response.status {response.status}")
                result.prompt = prompt
                result.success = False
                return
            chunks = []
            async for chunk, _ in response.content.iter_chunks():
                chunks.append(chunk)
        request_end_time = time.perf_counter()
        output = b"".join(chunks).decode("utf-8")
        output = json.loads(output)

    result.latency = request_end_time - request_start_time
    result.prompt = prompt
    result.response = output["choices"][0]["message"]["content"]
    result.num_prompt_tokens = output["usage"]["prompt_tokens"]
    result.num_completion_tokens = output["usage"]["completion_tokens"]
    result.energy = output["usage"]["energy"]

    pbar.update(1)


async def benchmark(
    results: Results,
    backend: str,
    model: str,
    api_url: str,
    input_requests: List[str],
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    pbar = tqdm(total=len(input_requests))
    async for i, request in get_request(input_requests, request_rate):
        prompt = request
        task = asyncio.create_task(
            # Ensures results has same ordering as the input dataset
            send_request(
                results.results[i],
                backend,
                model,
                api_url,
                prompt,
                pbar,
            )
        )
        tasks.append(task)
    await asyncio.gather(*tasks)
    pbar.close()


def run_benchmark(
    args: argparse.Namespace, api_url: str, input_requests: List[str], out_filename: str
):
    results = Results(
        model=args.model,
        backend=args.backend,
        request_rate=args.request_rate,
        results=[Result() for _ in input_requests],
    )

    zeus_monitor = ZeusMonitor()
    zeus_monitor.begin_window(out_filename)
    benchmark_start_time = time.perf_counter()
    asyncio.run(
        benchmark(
            results,
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

    # Store aggregated results
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency = 0
    total_latency_per_token = 0
    total_latency_per_output_token = 0
    server_total_energy = 0
    for result in results.results:
        if not result.success:
            results.num_failures += 1
            continue
        total_prompt_tokens += result.num_prompt_tokens
        total_completion_tokens += result.num_completion_tokens
        total_latency += result.latency
        total_latency_per_token += result.latency / (
            result.num_prompt_tokens + result.num_completion_tokens
        )
        total_latency_per_output_token += result.latency / result.num_completion_tokens
        server_total_energy += result.energy

    num_results = len(results.results) - results.num_failures
    if num_results == 0:
        print(f"{out_filename} not generated. All requests in this run failed.")
        return

    results.total_time = benchmark_end_time - benchmark_start_time
    results.throughput = num_results / results.total_time
    results.total_prompt_tokens = total_prompt_tokens
    results.total_completion_tokens = total_completion_tokens
    results.avg_latency = total_latency / num_results
    results.avg_latency_per_token = total_latency_per_token / num_results
    results.avg_latency_per_output_token = total_latency_per_output_token / num_results
    results.server_total_energy = server_total_energy
    results.server_energy_per_request = results.server_total_energy / num_results
    results.server_energy_per_output_token = (
        results.server_total_energy / results.total_completion_tokens
    )
    results.local_zeus_total_energy = zeus_total_energy
    results.local_zeus_energy_per_request = zeus_total_energy / num_results
    results.local_zeus_energy_per_output_token = (
        zeus_total_energy / results.total_completion_tokens
    )

    with open(out_filename, "w") as f:
        f.write(json.dumps(asdict(results), indent=2))

    if args.verbose:
        print("Benchmark results:")
        print(f"Model: {results.model}")
        print(f"Backend: {results.backend}")
        print(f"Request rate: {results.request_rate} requests/s")
        print()
        print(f"Total time: {results.total_time:.2f} s")
        print(f"Throughput: {results.throughput:.2f} requests/s")
        print(f"Average latency: {results.avg_latency:.2f} s")
        print(f"Average latency per token: {results.avg_latency_per_token:.2f} s")
        print(f"Average latency per output token: {results.avg_latency_per_output_token:.2f} s")
        print(f"(Zeus) Total energy: {results.local_zeus_total_energy:.2f} J")
        print(f"(Zeus) Energy per request: {results.local_zeus_energy_per_request:.2f} J")
        print(f"(Zeus) Energy per token: {results.local_zeus_energy_per_output_token:.2f} J")
        print(f"(Server) Total energy: {results.server_total_energy:.2f} J")
        print(f"(Server) Energy per request: {results.server_energy_per_request:.2f} J")
        print(f"(Server) Energy per token: {results.server_energy_per_output_token:.2f} J")

    print("Benchmark results written to", out_filename)


def main(args: argparse.Namespace):
    if args.backend not in ["tgi", "vllm"]:
        raise ValueError(f"Unknown backend: {args.backend}")

    arg_out_filename = f"{args.out_name}-args.json"
    with open(arg_out_filename, "w") as f:
        f.write(json.dumps(vars(args), indent=2))
    if args.verbose:
        print(args)
    print("Benchmark args written to", arg_out_filename)

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_name = args.out_name
    api_url = f"{args.protocol}://{args.host}:{args.port}{args.endpoint}"
    input_requests = get_requests(args.dataset)

    # Note: output filenames are 1-indexed
    for i in range(1, args.num_runs + 1):
        run_benchmark(args, api_url, input_requests, out_name + f"-run{i}.json")


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
        "--out-name",
        type=str,
        default="benchmark_result",
        help="Name of file to write benchmark results. Note: '-run{i}.json' will be appended for actual outputted files.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Set to true to print out benchmark results. Otherwise, only write to file.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
