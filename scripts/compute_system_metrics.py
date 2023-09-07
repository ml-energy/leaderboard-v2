import os
import csv
from glob import glob

import tyro
import pandas as pd


def main(data_dir: str, out_file: str) -> None:
    """Compute metrics for all models in the given directory."""
    model_names = os.listdir(data_dir)
    print(f"{model_names=}")

    if dirname := os.path.dirname(out_file):
        os.makedirs(dirname, exist_ok=True)
    out_csv = csv.writer(open(out_file, "w", newline=""))
    metrics = ["throughput", "response_length", "latency", "energy"]
    out_csv.writerow(["model", "batch_size"] + metrics)

    for model_name in model_names:
        for benchmark_file in glob(f"{data_dir}/{model_name}/benchmark_batch_*.json"):
            batch_size = int(benchmark_file.split("_")[-1][:-5])
            df = pd.read_json(benchmark_file)
            if len(df) < 2978 // batch_size and "Llama-2-7b" not in model_name:
                out_csv.writerow([model_name.replace("--", "/"), str(batch_size)] + ["OOM"])
            elif len(df) < (2978 // batch_size)*0.9 and "Llama-2-7b" in model_name: 
                out_csv.writerow([model_name.replace("--", "/"), str(batch_size)] + ["OOM"])
            else:
                out_csv.writerow(
                    [model_name.replace("--", "/"), str(batch_size)] + df[metrics].mean().to_list(),
                )


if __name__ == "__main__":
    tyro.cli(main)
