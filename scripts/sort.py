import json
import tyro

def main(data_dir:str, out_file:str) -> None:

    with open(data_dir, "r") as f:
        data = json.load(f)

    sorted_data = sorted(data, key=lambda x: len(x['conversations'][0]['value']), reverse=True)

    with open(out_file, "w") as f:
        json.dump(sorted_data, f, indent=4)

if __name__ == "__main__":
    tyro.cli(main)