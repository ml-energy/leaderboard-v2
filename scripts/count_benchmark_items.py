import os
import re
import json
import tyro
from collections import namedtuple

ModelBatch = namedtuple('ModelStruct', ['task','model_name', 'batch'])
model_state = {}

def load_and_print_length(task: str, root_dir: str) -> None:
    pattern = re.compile(r'benchmark_batch_(\d+).json')
    
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            match = pattern.match(file)
            if match:
                file_path = os.path.join(subdir, file)
                batch = int(match.group(1))
                print(f"Model: {os.path.basename(subdir)}, Batch size: {batch}")
                model_batch = ModelBatch(task=task, model_name=os.path.basename(subdir), batch=batch)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    print(f"Length: {len(data)}")
                    if len(data) >= 2000 / batch:
                        model_state[model_batch] = True
                    else:
                        model_state[model_batch] = False
                except json.JSONDecodeError:
                    print(f"[ERR] results found ")
                    model_state[model_batch] = False
                print("------")

def main(data_dir: str) -> None:
    """Summarize the results collected for all models in the given directory.
    
    Args:
        data_dir: The directory containing the results.
    """
    root_dir = ['chat', 'chat-concise', 'instruct', 'instruct-concise']
    for dir in root_dir:
        print(dir)
        load_and_print_length(dir, f"{data_dir}/{dir}")

    print("complete instance:")
    for info, stat in model_state.items():
        if stat is True:
            print(info)
    print("------")
    print("incomplete instance:")
    for info, stat in model_state.items():
        if stat is not True:
            print(info)

if __name__ == "__main__":
    tyro.cli(main)
