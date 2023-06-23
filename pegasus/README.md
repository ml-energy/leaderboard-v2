# Running benchmarks on multiple GPU nodes with Pegasus

[Pegasus](https://github.com/jaywonchung/pegasus) is an SSH-based multi-node command runner.
Different models have different verbosity, and benchmarking takes vastly different amounts of time.
Therefore, we want an automated piece of software that drains a queue of benchmarking jobs (one job per model) on a set of GPUs.

## Setup

### Install Pegasus

Pegasus needs to keep SSH connections with all the nodes in order to queue up and run jobs over SSH.
So you should install and run Pegasus on a computer that you can keep awake.

If you already have Rust set up:

```console
$ cargo install pegasus-ssh
```

Otherwise, you can set up Rust [here](https://www.rust-lang.org/tools/install), or just download Pegasus release binaries [here](https://github.com/jaywonchung/pegasus/releases/latest).

### Necessary setup for each node

Every node must have two things:

1. This repository cloned under `~/workspace/leaderboard`.
  - If you want a different path, search and replace in `setup-nodes.yaml`.
2. Model weights under `/data/leaderboard/weights`.
  - If you want a different path, search and replace in `setup-nodes.yaml` and `benchmark.yaml`.

### Specify node names for Pegasus

Modify `hosts.yaml` with nodes. See the file for an example.

- `hostname`: List the hostnames you would use in order to `ssh` into the node, e.g. `jaywonchung@gpunode01`.
- `gpu`: We want to create one Docker container for each GPU. List the indices of the GPUs you would like to use for the hosts.

### Set up Docker containers on your nodes with Pegasus

This builds our Docker image and spawns one container per GPU (named `leaderboard%d`), for every node.

```console
$ cd pegasus
$ cp setup-nodes.yaml queue.yaml
$ pegasus b
```

`b` stands for broadcast. Every command is run once on all (`hostname`, `gpu`) combinations.

## Benchmark

Now use Pegasus to run benchmarks for all the models across all nodes.

```console
$ cd pegasus
$ cp benchmark.yaml queue.yaml
$ pegasus q
```

`q` stands for queue. Each command is run once on the next available (`hostname`, `gpu`) combination.
