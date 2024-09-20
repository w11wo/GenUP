import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm.contrib.concurrent import thread_map

from gpt import GPT


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./data")
    parser.add_argument("--dataset", type=str, required=True, default="nyc")
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main(args):
    output_path = Path(args.output_path)

    dataset = load_dataset(args.dataset_id, split="train")

    llm = GPT()

    def generate_poi_reasoning(datum, index):
        output_file_path = output_path / args.dataset / "poi_reasoning" / f"poi_reasoning_{index}.json"
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        if output_file_path.exists():
            print(f"Index {index} already exists.")
            return

        # generate POI intention
        try:
            result = llm.generate_poi_intention(**datum)
        except json.decoder.JSONDecodeError:
            print(f"Index {index} failed.")
            return

        # save POI intention
        with open(output_file_path, "w") as f:
            json.dump(result, f, indent=4)

    _ = thread_map(generate_poi_reasoning, dataset, range(len(dataset)), max_workers=args.num_workers)


if __name__ == "__main__":
    args = parse_args()
    main(args)
