import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from datasets import Dataset
from tqdm.contrib.concurrent import thread_map

from gpt import GPT


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./LLM4POI/")
    parser.add_argument("--output_path", type=str, default="./data")
    parser.add_argument("--dataset", type=str, required=True, default="nyc")
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def create_checkin_prompt(row):
    prompt = "At {UTCTimeOffset}, user {UserId} visited POI id {PoiId} which is a/an {PoiCategoryName} with category id {PoiCategoryId}."
    return prompt.format(**row)


def main(args):
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    dataset_path = input_path / args.dataset / "preprocessed"

    # only generate user summary from training data
    train_df = pd.read_csv(dataset_path / "train_sample.csv")
    # create check-in prompt similar to LLM4POI
    train_df.loc[:, "prompt"] = train_df.apply(create_checkin_prompt, axis=1)

    users = train_df["UserId"].unique()

    llm = GPT()

    def generate_user_profile(user_id):
        output_file_path = output_path / args.dataset / "user_profiles" / f"user_profile_{user_id}.json"
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        if output_file_path.exists():
            print(f"User {user_id} profile already exists.")
            return

        user_df = train_df[train_df["UserId"] == user_id]
        user_history_prompt = " ".join(user_df["prompt"].values)

        # generate user profile summary
        result = llm.generate_user_profile(user_id=user_id, user_history_prompt=user_history_prompt)

        # save user profile summary
        with open(output_file_path, "w") as f:
            json.dump(result, f, indent=4)

    _ = thread_map(generate_user_profile, users, max_workers=args.num_workers)

    json_files = sorted(
        (output_path / args.dataset / "user_profiles").glob("user_profile_*.json"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )

    user_profiles = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            user_profile = json.load(f)
        user_profile["user_id"] = json_file.stem.split("_")[-1]
        user_profile["attributes"] = list(map(str, user_profile["attributes"]))
        user_profiles.append(user_profile)

    profiles_df = pd.DataFrame(user_profiles)
    profiles_df = profiles_df[["user_id", "user_profile", "traits", "attributes", "preferences", "routines"]]
    profiles_ds = Dataset.from_pandas(profiles_df)
    profiles_ds.push_to_hub(args.dataset_id)


if __name__ == "__main__":
    args = parse_args()
    main(args)
