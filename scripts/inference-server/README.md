# About

This directory contains a script for running benchmarks (including energy comsumption) on models that are hosted on a dedicated inference server. The script is taken and modified from [vllm](https://github.com/vllm-project/vllm/blob/93b38bea5dd03e1b140ca997dfaadef86f8f1855/benchmarks/benchmark_serving.py)

The current script supports TGI and vLLM. Before running the benchmark script, the inference server hosting the relevant model should be hosted. 
