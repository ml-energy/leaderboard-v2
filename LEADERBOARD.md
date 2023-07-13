The goal of the ML.ENERGY Leaderboard is to give people a sense of how much **energy** LLMs would consume.

The code for the leaderboard, backing data, and scripts for benchmarking are all open-source in our [repository](https://github.com/ml-energy/leaderboard).
We'll see you at the [Discussion board](https://github.com/ml-energy/leaderboard/discussions), where you can ask questions, suggest improvement ideas, or just discuss leaderboard results!

## Columns

- `gpu`: NVIDIA GPU model name.
- `task`: Name of the task. See *Tasks* below for details.
- `energy` (J): The average GPU energy consumed by the model to generate a response.
- `throughput` (token/s): The average number of tokens generated per second.
- `latency` (s): The average time it took for the model to generate a response.
- `response_length` (token): The average number of tokens in the model's response.
- `parameters`: The number of parameters the model has, in units of billion.
- `arc`: [AI2 Reasoning Challenge](https://allenai.org/data/arc)'s `challenge` dataset. Measures capability to do grade-school level question answering, 25 shot.
- `hellaswag`: [HellaSwag dataset](https://allenai.org/data/hellaswag). Measuring grounded commonsense, 10 shot.
- `truthfulqa`: [TruthfulQA dataset](https://arxiv.org/abs/2109.07958). Measuring truthfulness against questions that elicit common falsehoods, 0 shot.

NLP evaluation metrics (`arc`, `hellaswag`, and `truthfulqa`) were only run once each on A40 GPUs because their results do not depend on the GPU type.
Hence, all GPU model rows for the same model share the same NLP evaluation numbers.

## Tasks

For each task, every model uses the same system prompt. We still account for differences in roles, e.g. `USER`, `HUMAN`, `ASSISTANT`, `GPT`.

| Name | System prompt |
|--|--|
| chat | A chat between a human user (prompter) and an artificial intelligence (AI) assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. |
| chat-concise | A chat between a human user (prompter) and an artificial intelligence (AI) assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant's answers are very concise. |
| instruct | Below is an instruction that describes a task. Write a response that appropriately completes the request. |
| instruct-concise | Below is an instruction that describes a task. Write a response that appropriately completes the request. The response should be very concise. |

You can see that response length is shorter on average for the `-concise` variants of the tasks.
This affects the number of decoding iterations the model has to run in order to finish responding, thus affecting latency and energy consumption per prompt.

## Setup

Find our benchmark script for one model [here](https://github.com/ml-energy/leaderboard/blob/master/benchmark.py).

### Software

- PyTorch 2.0.1
- [Zeus](https://ml.energy/zeus) -- For GPU time and energy measurement
- [FastChat](https://github.com/lm-sys/fastchat) -- For running inference on various models
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/72b7f0c00a6ff94632c5b873fc24e093ae74fa47) -- For NLP evaluation metrics

### Hardware

- NVIDIA A40 GPU
- NVIDIA A100 GPU
- NVIDIA V100 GPU

### Parameters

- Model
  - Batch size 1
  - FP16
- Sampling (decoding)
  - Greedy sampling from multinomial distribution
  - Temperature 0.7
  - Repetition penalty 1.0

### Data

We randomly sampled around 3000 prompts from the [cleaned ShareGPT dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered).
See [here](https://github.com/ml-energy/leaderboard/tree/master/sharegpt) for more detail on how we created the benchmark dataset.

## FAQ

### So who's the winner?

It depends on which metric you value most.
Some may be tightly constrained by electricity consumption, in which case energy would have higher weight.
Some may just want better model quality, in which case the NLP dataset results will be important.
Others might want something balanced.
This is why we support adding custom columns to the table, and let you choose your own winner!

### Where can I find more about ML energy-related resources?

Meet us at the [ML.ENERGY initiative](https://ml.energy) homepage!

## Contributing

Any kind of contribution is more than welcome!
Please look around our [repository](https://github.com/ml-energy/leaderboard).

Especially, if you want to see a specific model on the leaderboard, please consider adding support to the model.
We'll consider running those on the hardware we have.
First, see if the model is available in Hugging Face Hub and compatible with lm-evaluation-harness.
Then, in our [`benchmark.py`](https://github.com/ml-energy/leaderboard/blob/master/scripts/benchmark.py), implement a way to load the weights of the model and run generative inference.

Currently, we use FastChat to load models and run inference, but we'll eventually abstract the model executor, making it easier to add models that FastChat does not support.

## Limitations

Currently, inference is run with basically bare PyTorch with batch size 1, which is unrealistic assuming a production serving scenario.
Hence, absolute latency, throughput, and energy numbers should not be used to estimate figures in real production settings, while relative comparison makes some sense.

Batch size 1, in some sense, is the lowest possible hardware utilization.
We'll soon benchmark batch sizes larger than 1 without continuous batching for comparison.
This would show what happens in the case of very high hardware utilization (although it's with PyTorch), assuming an ideal case where all sequences in each batch generate the same number of output tokens.
By doing this, we can provide numbers for reasonable comparison without being tied to any existing generative model serving system.

## Upcoming

- Within the Summer, we'll add an online text generation interface for real time energy consumption measurement!
- Batched inference
- More optimized inference runtimes, like TensorRT.
- Larger models with distributed inference, like Falcon 40B.
- More models, like RWKV.

## License

This leaderboard is a research preview intended for non-commercial use only.
Model weights were taken as is from the Hugging Face Hub if available and are subject to their licenses.
The use of LLaMA weights are subject to their [license](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md).
Please direct inquiries/reports of potential violation to Jae-Won Chung.

## Acknowledgements

We thank [Chameleon Cloud](https://www.chameleoncloud.org/) and [CloudLab](https://cloudlab.us/) for the GPU nodes.
