"""Taken and modified from vllm: https://github.com/vllm-project/vllm/blob/93b38bea5dd03e1b140ca997dfaadef86f8f1855/benchmarks/benchmark_serving.py
   Filter dataset to:
   1. Remove entries that have too long prompts or completions
   2. Only keep first human prompt for each conversation
"""

import json
import random
from typing import AsyncGenerator, List, Tuple

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

def filter_dataset_to_size(
    dataset_path: str,
    size: int,
) ->  List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    # randomly sample dataset
    return random.sample(dataset, size)


def filter_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (
            data["id"],
            data["conversations"][0]["value"],
            data["conversations"][1]["value"],
        )
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    conversation_ids = [conv_id for conv_id, _, _ in dataset]
    prompts = [prompt for _, prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append(
            (conversation_ids[i], prompts[i], prompt_token_ids[i], output_len)
        )

    # Filter out too long sequences.
    filtered_dataset_json = []
    for conv_id, prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        # making even shorter than 1024 to account for additional tokens introduced by chat completion wrapper
        if prompt_len > 800 or output_len > 800:
        # if prompt_len > 1024 or output_len > 1024:
            # Prune too long sequences.
            continue
        filtered_dataset_json.append(
            {
                "id": conv_id,
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt,
                    }
                ],
            }
        )

    return filtered_dataset_json


def main():
    # tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    # new_dataset = filter_dataset(
    #     "ShareGPT_V3_unfiltered_cleaned_split.json", tokenizer
    # )
    # with open("ShareGPT_V3_filtered.json", "w") as f:
    #     json.dump(new_dataset, f)

    new_dataset = filter_dataset_to_size(
        "ShareGPT_V3_filtered.json", 1000
    )
    with open("ShareGPT_V3_filtered_1000.json", "w") as f:
        json.dump(new_dataset, f)

    print(f"New dataset size: {len(new_dataset)}")


if __name__ == "__main__":
    main()
