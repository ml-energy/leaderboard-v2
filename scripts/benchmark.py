"""Perform inference of one model on a dataset and measure time and energy."""

from __future__ import annotations

import os
import json
import copy
import atexit
from typing import Generator, Literal, Iterable, Dict
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

import gc
import numpy as np
import tyro
import torch
import rich
from rich.table import Table
from fastchat.serve.inference import prepare_logits_processor
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

@dataclass
class Output:
    response_length: int
    input: str
    output: str

@torch.inference_mode()
def run_inference(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int = 2048,
) ->list[Output]:
    # Read parameters
    prompts = params["prompt"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    stop_token_ids = list(params.get("stop_token_ids", None) or [])
    stop_token_ids.append(tokenizer.eos_token_id)
    batch_size = len(prompts)

    empty_result = Output(response_length=-1, input="", output="")
    result = []
    for i, prompt in enumerate(prompts):
        result.append(copy.deepcopy(empty_result))
        result[i].input = prompt

    # left append prompts with eos to make all input prompts the same length
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer(prompts, padding=True).input_ids
    output_ids = [[] for _ in range(batch_size)]

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:  # truncate
        max_src_len = context_len - max_new_tokens - 1

    input_ids = [input_id[-max_src_len:] for input_id in input_ids]

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor(input_ids, device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id] for _ in range(batch_size)],
            dtype=torch.int64,
            device=device,
        )
    
    past_key_values = out = None
    stopped = np.array(batch_size*[False])
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor(input_ids, device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:  # decoding
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [[token[0]] for token in tokens], device=device
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [[token[0]] for token in tokens], device=device
                    ),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor(output_ids, device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])
        else:
            last_token_logits = logits[:, -1, :]

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [[int(token) for token in query] for query in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [[int(token) for token in query] for query in indices.tolist()]

        output_ids = [ids + [token[0]] for ids, token in zip(output_ids, tokens)]
        
        # deal with stop_token_ids
        old_stopped = stopped
        stopped = np.logical_or(old_stopped, np.array([True if token[0] in stop_token_ids else False for token in tokens]))
        different_indices = np.where(stopped != old_stopped)[0]

        rfind_start = 0
        output = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        output_np = np.array(output)

        if different_indices.size > 0:
            # here i but not i+1 is because the i+1 token generated is in stop_token_ids
            for j in different_indices:
                result[j].response_length = i
                result[j].output = output[j]

        # deal with stop_str
        if stop_str:
            if isinstance(stop_str, str):
                pos_array = np.char.rfind(output_np, stop_str, rfind_start)
                find_stop = pos_array != -1
            elif isinstance(stop_str, Iterable):
                for each_stop in stop_str:
                    pos_array = np.char.rfind(output_np, each_stop, rfind_start)
                    find_stop = pos_array != -1
            else:
                raise ValueError("Invalid stop field type.")

            stop_str_indices = np.where(find_stop & ~stopped)[0]
            if stop_str_indices.size > 0:
                for j in stop_str_indices:
                    # TODO: find a elegant way to figure out the size of stop_str, here just suppose stop_str has one token
                    result[j].response_length = i
                    result[j].output = output[j][:pos_array[j]]
                stopped[find_stop] = True

        if all(stopped):
            break

    not_finish_indices = np.where(stopped == False)[0]
    if any(stopped) == False:
        output = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        for j in not_finish_indices:
            result[j].response_length = max_new_tokens
            result[j].output = output[j]

    return result


def main(
    model_path: str,
    input_file: str = "sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled_sorted.json",
    output_dir: str = "data",
    device_index: int = 0,
    task: Literal[tuple(SYSTEM_PROMPTS)] = "chat",  # type: ignore
    load_8bit: bool = False,
    temperature: float = 0.7,
    repitition_penalty: float = 1.0,
    max_new_tokens: int = 512,
    batch: int = 1,
) -> None:
    """Run benchmarking for one model on the entire input file.

    Args:
        model_path: Path to or Huggingface Hub Id of the model.
        input_file: Path to the input JSON file. Assumed to be our cleaned ShareGPT data.
            (Default: "sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled_sorted.json")
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
    output_csv_path = f"{output_dir}/benchmark_batch_{batch}.json"
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
    if "llama-2" in model_path.lower():
        conv_base.system = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPTS[task]}\n<</SYS>>\n\n"
    elif "stablelm" in model_path.lower():
        conv_base.system = f"""<|SYSTEM|># {SYSTEM_PROMPTS[task]}\n"""
    else:
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
                "batch_size": batch,
            },
            config_json,
            indent=4,
        )
        config_json.write("\n")

    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            sample = self.data[index]
            return sample["conversations"][0]["value"]


    def dataloader(input_file: str, batch_size: batch) -> Generator[tuple[bool, str], None, None]:
        """Yields a tuple of whether this is a warmup run and the input prompt."""
        for _ in range(3):
            yield True, ["Say something long and random. I don't care about the content." for _ in range (batch)]
        data = json.load(open(input_file, "r"))
        custom_dataset = CustomDataset(data)
        data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
        for prompt in data_loader:
            yield False, prompt

    # Warm up the GPU with some random prompts.
    # Forward through all the prompts.
    is_first = True
    convs = []
    prompts = []
    data_iter = iter(dataloader(input_file, batch))
    
    for is_warmup, input_prompts in data_iter:
        # Construct the input prompt.
        for i in range(batch):
            conv = copy.deepcopy(conv_base)
            conv.append_message(conv.roles[0], input_prompts[i])
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()
            prompts.append(prompt)
            convs.append(conv)

        gen_params["prompt"] = prompts

        # Print input prompt.
        for i in range(len(convs)):
            console.print(f"\n[u cyan]{'Warmup ' if is_warmup else ''}Prompt[/u cyan](batch_{i}):")
            console.print(prompts[i].strip() + "\n", markup=False)

        #################################################
        # Inference and measurement zone!
        #################################################
        monitor.begin_window("inference")
        results = run_inference(model, tokenizer, gen_params, device="cuda", context_len=2048)
        measurements = monitor.end_window("inference")
        #################################################
        
        # Record numbers.
        if not is_warmup:
            response_length = sum([result.response_length for result in results])  # number of valid tokens
            latency = measurements.time
            throughput = response_length / latency
            energy = measurements.total_energy
            output = {
                "model": model_name_cleaned,
                "throughput": throughput,
                "response_length": response_length,
                "latency": latency,
                "energy": energy,
                "input": [prompt.strip() for prompt in prompts],
                "output": [(result.output).strip() for result in results],
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
        for i in range(len(convs)):
            console.print(f"\n[u cyan]{'Warmup ' if is_warmup else ''}Response[/u cyan](batch_{i}):")
            console.print(results[i].output.strip() + "\n", markup=False)

        # Print measurement.
        console.print(measurements)
        convs = []
        prompts = []

if __name__ == "__main__":
    tyro.cli(main)
