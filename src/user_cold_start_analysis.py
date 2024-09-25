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
    dataset_id = args.dataset_id.split("/")[-1]

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # get user ID from inputs
    get_user_id = lambda x: re.match(r"The following data is a trajectory of user (\d+):.*", x).group(1)
    train_df["user"] = train_df["inputs"].apply(get_user_id)
    test_df["user"] = test_df["inputs"].apply(get_user_id)

    # get number of trajectories per user
    user_trajectories = train_df["user"].value_counts().sort_values(ascending=False)

    # split users into three categories: very active, normal, inactive
    # very active users are the top 1/3 of users by number of trajectories
    # inactive users are the bottom 1/3 of users by number of trajectories
    # normal users are the rest
    n = len(user_trajectories) // 3
    very_active_users = user_trajectories.nlargest(n).index
    inactive_users = user_trajectories.nsmallest(n).index

    train_df["user_activity"] = "normal"
    train_df.loc[train_df["user"].isin(very_active_users), "user_activity"] = "very_active"
    train_df.loc[train_df["user"].isin(inactive_users), "user_activity"] = "inactive"

    user2activity = dict(zip(train_df["user"], train_df["user_activity"]))

    # for test_df, get user activity from train_df
    test_df["user_activity"] = test_df["user"].apply(lambda user: user2activity[user])

    # load eval results
    with open(f"results/{model_checkpoint}-{dataset_id}.json") as f:
        results = json.load(f)

    # add predictions and labels to test_df
    test_df["predictions"] = results["predictions"]
    test_df["labels"] = results["targets"]

    # calculate accuracy for each user activity category
    accuracies = {}

    for user_activity in ["inactive", "normal", "very_active"]:
        _test_df = test_df[test_df["user_activity"] == user_activity]
        accuracies[user_activity] = accuracy_score(_test_df["labels"].to_list(), _test_df["predictions"].to_list())

    # save results
    results["user_cold_start_analysis"] = accuracies

    with open(f"results/{model_checkpoint}-{dataset_id}.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
