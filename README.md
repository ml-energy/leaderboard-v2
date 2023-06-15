---
title: "ML.ENERGY Leaderboard"
python_version: "3.9"
app_file: "app.py"
sdk: "gradio"
sdk_version: "3.23.0"
pinned: false
---

# ML.ENERGY Leaderboard

## Devs

Currently setup in `ampere02`:

1. Find model weights in `/data/leaderboard/weights/`, e.g. subdirectory `llama` and `vicuna`.

2. Let's share the Huggingface Transformer cache:

```bash
export TRANSFORMERS_CACHE=/data/leaderboard/hfcache
```
