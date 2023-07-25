---
title: "ML.ENERGY Leaderboard"
emoji: "⚡"
python_version: "3.9"
app_file: "app.py"
sdk: "gradio"
sdk_version: "3.35.2"
pinned: true
tags: ["energy", "leaderboard"]
colorFrom: "black"
colorTo: "black"
---

# ML.ENERGY Leaderboard

[![Leaderboard](https://custom-icon-badges.herokuapp.com/badge/ML.ENERGY-Leaderboard-blue.svg?logo=ml-energy)](https://ml.energy/leaderboard)
[![Deploy](https://github.com/ml-energy/leaderboard/actions/workflows/push_spaces.yaml/badge.svg?branch=web)](https://github.com/ml-energy/leaderboard/actions/workflows/push_spaces.yaml)
[![Apache-2.0 License](https://custom-icon-badges.herokuapp.com/github/license/ml-energy/leaderboard?logo=law)](/LICENSE)

How much energy do LLMs consume?

This README focuses on explaining how to run the benchmark yourself.
The actual leaderboard is here: https://ml.energy/leaderboard.

## Setup

### Model weights

- For models that are directly accessible in Hugging Face Hub, you don't need to do anything.
- For other models, convert them to Hugging Face format and put them in `/data/leaderboard/weights/lmsys/vicuna-13B`, for example. The last two path components (e.g., `lmsys/vicuna-13B`) are taken as the name of the model.

### Docker container

We have our pre-built Docker image published with the tag `mlenergy/leaderboard:latest` ([Dockerfile](/Dockerfile)).

```console
$ docker run -it \
    --name leaderboard0 \
    --gpus '"device=0"' \
    -v /path/to/your/data/dir:/data/leaderboard \
    -v $(pwd):/workspace/leaderboard \
    mlenergy/leaderboard:latest bash
```

The container internally expects weights to be inside `/data/leaderboard/weights` (e.g., `/data/leaderboard/weights/lmsys/vicuna-7B`), and sets the Hugging Face cache directory to `/data/leaderboard/hfcache`.
If needed, the repository should be mounted to `/workspace/leaderboard` to override the copy of the repository inside the container.

## Running the benchmark

We run benchmarks using multiple nodes and GPUs using [Pegasus](https://github.com/jaywonchung/pegasus). Take a look at [`pegasus/`](/pegasus) for details.

You can still run benchmarks without Pegasus like this:

```console
$ docker exec leaderboard0 python scripts/benchmark.py --model-path /data/leaderboard/weights/lmsys/vicuna-13B --input-file sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled_sorted.json
$ docker exec leaderboard0 python scripts/benchmark.py --model-path databricks/dolly-v2-12b --input-file sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled_sorted.json
```
