"""Perform inference of one model on a dataset and measure time and energy."""

from __future__ import annotations

import os
import json
import copy
import atexit
from typing import Generator, Literal

import tyro
import torch
import rich
from rich.table import Table
from fastchat.serve.inference import generate_stream
from fastchat.model.model_adapter import load_model, get_conversation_template
from zeus.monitor import ZeusMonitor

SYSTEM_PROMPTS = {
    "chat": (
        "A chat between a human user (prompter) and an artificial intelligence (AI) assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    ),
    "chat-concise": (
        "A chat between a human user (prompter) and an artificial intelligence (AI) assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "The assistant's answers are very concise. "
    ),
    "instruct": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request. "
    ),
    "instruct-concise": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request. "
        "The response should be very concise. "
    ),
}


def main(
    model_path: str,
    input_file: str = "sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json",
    output_dir: str = "data",
    device_index: int = 0,
    task: Literal[tuple(SYSTEM_PROMPTS)] = "chat",  # type: ignore
    load_8bit: bool = False,
    temperature: float = 0.7,
    repitition_penalty: float = 1.0,
    max_new_tokens: int = 512,
) -> None:
    """Run benchmarking for one model on the entire input file.

    Args:
        model_path: Path to or Huggingface Hub Id of the model.
        input_file: Path to the input JSON file. Assumed to be our cleaned ShareGPT data.
            (Default: "sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json")
        output_dir: Path to the output directory. (Default: "data")
        device_index: Index of the GPU to use for inference. (Default: 0)
        task: Type of task to perform inference on. (Default: "chat")
        load_8bit: Whether to load the model in 8-bit mode. (Default: False)
        temperature: Temperature to use for sampling. (Default: 0.7)
        repitition_penalty: Repitition penalty to use for the model. (Default: 1.0)
        max_new_tokens: Maximum numbers of tokens to generate, ignoring the prompt. (Default: 512)
    """
    # NOTE(JW): ChatGLM is implemented as a special case in FastChat inference.
    # Also, it's primarily a model that's fine-tuned for Chinese, so it doesn't
    # make sense to prompt it in English and talk about its verbosity.
    if "chatglm" in model_path.lower():
        raise ValueError("ChatGLM is not supported.")

    # Get Rich Console instance.
    console = rich.get_console()

    # Print out what we're about to do.
    if model_path.endswith("/"):
        model_path = model_path[:-1]
    model_name_cleaned = "--".join(model_path.split("/")[-2:])
    output_dir = f"{output_dir}/{task}/{model_name_cleaned}"
    output_csv_path = f"{output_dir}/benchmark.json"
    config_json_path = f"{output_dir}/config.json"
    table = Table(title="Benchmark")
    table.add_column("Configuration")
    table.add_column("Value")
    table.add_row("Model", f"{model_name_cleaned} (path: {model_path})")
    table.add_row("Input", input_file)
    table.add_row("Device", f"cuda:{device_index}")
    table.add_row("Task", task)
    table.add_row("8-bit", str(load_8bit))
    table.add_row("Temperature", f"{temperature:.2f}")
    table.add_row("Repitition Penalty", f"{repitition_penalty:.2f}")
    table.add_row("Max New Tokens", str(max_new_tokens))
    table.add_row("Output CSV", output_csv_path)
    table.add_row("Config JSON", config_json_path)
    console.print(table)

    # Set the device.
    torch.cuda.set_device(f"cuda:{device_index}")

    # Load the model (Huggingface PyTorch) and tokenizer (Huggingface).
    model, tokenizer = load_model(
        model_path=model_path,
        device="cuda",
        num_gpus=1,
        max_gpu_memory=None,
        load_8bit=load_8bit,
        cpu_offloading=False,
        gptq_config=None,
        debug=False,
    )

    # Chats are accumulated in a conversation helper object.
    conv_base = get_conversation_template(model_path)

    # Standardize the system prompt for every model.
    conv_base.system = SYSTEM_PROMPTS[task]
    conv_base.messages = []
    conv_base.offset = 0

    gen_params = {
        "model": model_path,
        "prompt": "EMPTY",
        "temperature": temperature,
        "repitition_penalty": repitition_penalty,
        "max_new_tokens": max_new_tokens,
        "stop": conv_base.stop_str,
        "stop_token_ids": conv_base.stop_token_ids,
        "echo": False,
    }

    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

    # Output files.
    # Leave only the last two path components and replace slashes with double dashes.
    os.makedirs(output_dir, exist_ok=True)
    output_json = open(output_csv_path, "w")
    output_json.write("[\n")
    output_json.flush()
    # Conclude the JSON file format with a closing bracket. Using `atexit` will
    # handle all cases of the program exiting, including Ctrl-C and errors.
    atexit.register(lambda: output_json.write("\n]\n"))

    # Dump the configuration to a JSON file.
    with open(config_json_path, "w") as config_json:
        json.dump(
            {
                "model_path": model_path,
                "input_file": input_file,
                "device_index": device_index,
                "task": task,
                "load_8bit": load_8bit,
                "temperature": temperature,
                "repitition_penalty": repitition_penalty,
                "max_new_tokens": max_new_tokens,
            },
            config_json,
            indent=4,
        )
        config_json.write("\n")

    def dataloader(input_file: str) -> Generator[tuple[bool, str], None, None]:
        """Yields a tuple of whether this is a warmup run and the input prompt."""
        for _ in range(3):
            yield True, "Say something long and random. I don't care about the content."
        for item in json.load(open(input_file, "r")):
            input_prompt = item["conversations"][0]["value"]
            yield False, input_prompt

    # Warm up the GPU with some random prompts.
    # Forward through all the prompts.
    is_first = True
    for is_warmup, input_prompt in dataloader(input_file):
        # Construct the input prompt.
        conv = copy.deepcopy(conv_base)
        conv.append_message(conv.roles[0], input_prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        gen_params["prompt"] = prompt

        # Print input prompt.
        console.print(f"\n[u cyan]{'Warmup ' if is_warmup else ''}Prompt[/u cyan]:")
        console.print(prompt.strip() + "\n", markup=False)

        # Generate the ouptut from the model.
        output_stream = generate_stream(model, tokenizer, gen_params, device="cuda")
        output = {}

        #################################################
        # Inference and measurement zone!
        #################################################
        monitor.begin_window("inference")
        for output in output_stream:
            pass
        measurements = monitor.end_window("inference")
        #################################################
        
        # Record numbers.
        output_text = output["text"]
        if not is_warmup:
            response_length = len(tokenizer.encode(output_text))  # number of tokens
            latency = measurements.time
            throughput = response_length / latency
            energy = measurements.total_energy
            output = {
                "model": model_name_cleaned,
                "throughput": throughput,
                "response_length": response_length,
                "latency": latency,
                "energy": energy,
                "input": prompt.strip(),
                "output": output_text.strip(),
            }
            output_str = json.dumps(output, indent=4)
            if not is_warmup:
                if not is_first:
                    output_json.write(",\n" + output_str)
                else:
                    is_first = False
                    output_json.write(output_str)
            output_json.flush()

        # Print the response.
        console.print(f"\n[u cyan]{'Warmup ' if is_warmup else ''}Response[/u cyan]:")
        console.print(output_text.strip() + "\n", markup=False)

        # Print measurement.
        console.print(measurements)


if __name__ == "__main__":
    tyro.cli(main)
