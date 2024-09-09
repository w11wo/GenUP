import re
import json

from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
from datasets import load_dataset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    model_checkpoint = args.model_checkpoint.split("/")[-1]

    dataset = load_dataset(args.dataset_id)

    test_df = dataset["test"].to_pandas()

    # get trajectory length for each trajectory
    test_df["trajectory_length"] = test_df["inputs"].apply(lambda x: len(re.findall(r"user \d+ visited POI id", x)))

    # split trajectory lengths into three categories: short, middle, long
    # long trajectories are the top 1/3 of trajectory lengths
    # short trajectories are the bottom 1/3 of trajectory lengths
    n = len(test_df) // 3
    long_trajectories = test_df["trajectory_length"].nlargest(n).values
    short_trajectories = test_df["trajectory_length"].nsmallest(n).values

    test_df["trajectory_type"] = "middle"
    test_df.loc[test_df["trajectory_length"].isin(long_trajectories), "trajectory_type"] = "long"
    test_df.loc[test_df["trajectory_length"].isin(short_trajectories), "trajectory_type"] = "short"

    # load eval results
    with open(f"results/{model_checkpoint}.json") as f:
        results = json.load(f)

    # add predictions and labels to test_df
    test_df["predictions"] = results["predictions"]
    test_df["labels"] = results["targets"]

    # calculate accuracy for each trajectory type
    accuracies = {}

    for trajectory_type in ["short", "middle", "long"]:
        _test_df = test_df[test_df["trajectory_type"] == trajectory_type]
        accuracies[trajectory_type] = accuracy_score(_test_df["labels"].to_list(), _test_df["predictions"].to_list())

    # save results
    results["trajectory_length_analysis"] = accuracies

    with open(f"results/{model_checkpoint}.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
