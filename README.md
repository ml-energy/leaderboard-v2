---
title: "ML.ENERGY Leaderboard"
python_version: "3.9"
app_file: "app.py"
sdk: "gradio"
sdk_version: "3.35.2"
pinned: true
---

# ML.ENERGY Leaderboard

The goal of the ML.ENERGY Leaderboard is to give people a sense of how much **energy** LLMs would consume.

## How is energy different?

Even between models with the exact same architecture and size, the average energy consumption per prompt is different because they have **different verbosity**.
That is, when asked the same thing, they answer in different lengths.

## Metrics

- `lmsys_elo`: The ELO score from the LMSys leaderboard. Thanks!
- `throughput` (token/s): The average number of tokens generated per second.
- `response_length` (token): The average number of tokens in the model's response.
- `latency` (s): The average time it took for the model to generate a response.
- `energy` (J): The average energy consumed by the model to generate a response.

## Setup

Find our benchmark script for one model [here](https://github.com/ml-energy/leaderboard/blob/master/benchmark.py).

### Software

- PyTorch 2.0.1
- [FastChat](https://github.com/lm-sys/fastchat) -- For various model support
- [Zeus](https://ml.energy/zeus) -- For GPU energy measurement

### Hardware

- NVIDIA A40 GPU

### Parameters

- Model
  - Batch size 1
  - FP16
- Sampling (decoding)
  - Greedy sampling from multinomial distribution
  - Temperature 0.7
  - Repetition penalty 1.0

## Data

We randomly sampled around 3000 prompts from the [cleaned ShareGPT dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered).
See [here](https://github.com/ml-energy/leaderboard/tree/master/sharegpt) for more detail on how we created the benchmark dataset.

We used identical system prompts for all models (while respecting their own *role* tokens):
```
A chat between a human user (prompter) and an artificial intelligence (AI) assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
```

## Upcoming

- Compare against more optimized inference runtimes, like TensorRT.
- Other GPUs
- Other model/sampling parameters
- More models
- Model quality evaluation numbers (e.g., AI2 Reasoning Challenge, HellaSwag)
