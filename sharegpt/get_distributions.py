import json
import matplotlib.pyplot as plt
import numpy as np
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

# Open datasets
file_paths = ['ShareGPT_V3_filtered.json', 'ShareGPT_V3_filtered_1000.json']
names = [file_path[:-5] for file_path in file_paths]

data_lists =[]
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
        data_lists.append(data_list)
print('Dataset sizes:')
for name, data_list in zip(names, data_lists):
    print(f'{name}: {len(data_list)}')

# Get prompt lengths using tokenizer
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
all_prompts = [[data['conversations'][0]['value'] for data in data_lists] for data_lists in data_lists]
all_token_ids_per_prompts = [tokenizer(prompts).input_ids for prompts in all_prompts]
all_prompt_lens = [[len(token_ids) for token_ids in token_ids_per_prompt] for token_ids_per_prompt in all_token_ids_per_prompts]

# Plotting the histograms
for name, prompt_lens in zip(names, all_prompt_lens):
    plt.hist(prompt_lens, bins=range(min(prompt_lens), max(prompt_lens) + 1), edgecolor='black')
    plt.xlabel('Prompt Length (number of tokens)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {name}')
    plt.savefig(f'{name}_distribution.png')
    plt.close()

# Plotting the CDF
for name, prompt_lens in zip(names, all_prompt_lens):
    values, counts = np.unique(prompt_lens, return_counts=True)
    relative_frequencies = counts / len(prompt_lens)
    sorted_data = np.sort(values)
    cumulative_frequencies = np.cumsum(relative_frequencies)
    plt.step(sorted_data, cumulative_frequencies, where='post', label=name)

plt.title(f'Cumulative Distribution Function (CDF) Overlayed')
plt.xlabel('Prompt Length (number of tokens)')
plt.ylabel('Cumulative Probability')
plt.savefig(f'{name}_cdf.png')
plt.close()
