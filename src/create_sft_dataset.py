import re
import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict

LLAMA_PROMPT = """<s>[INST] <<SYS>>
{profile}
<</SYS>>

{instruction} [/INST] {answer} </s>"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, default="../LLM4POI/datasets/")
    parser.add_argument("--profiles_path", type=str, default="./data")
    parser.add_argument("--dataset", type=str, required=True, default="nyc")
    parser.add_argument("--dataset_id", type=str, required=True)
    return parser.parse_args()


def create_system_prompt(user_profile: dict, user_id: str) -> str:
    age, gender, education, socioeco = user_profile["attributes"]
    traits = ", ".join(user_profile["traits"])
    preferences = ", ".join(user_profile["preferences"])
    routines = ", ".join(user_profile["routines"])
    user_profile_str = user_profile["user_profile"]
    system_prompt = f"""You are user {user_id} and your basic information is as follows:
Age: {age}; Gender: {gender}; Education: {education}; SocioEco: {socioeco}.
You have the following traits: {traits}.
You have the following preferences: {preferences}.
You have the following routines: {routines}.
{user_profile_str}"""
    return system_prompt


def get_user_profile(user_id: str, profiles_path: Path) -> dict:
    user_profile_file = profiles_path / f"user_profile_{user_id}.json"
    with user_profile_file.open() as f:
        user_profile = json.load(f)
    return user_profile


def main(args):
    input_path = Path(args.input_path)
    dataset_path = input_path / args.dataset / "preprocessed"

    profiles_path = Path(args.profiles_path) / args.dataset

    # load original QA pairs
    with open(dataset_path / "train_qa_pairs_kqt.json") as f:
        raw_data_train = json.load(f)

    with open(dataset_path / "test_qa_pairs_kqt.txt") as f:
        raw_data_test = f.read().splitlines()

    # convert test data to dict format
    raw_data_test = [{"question": q, "answer": a} for line in raw_data_test for (q, a) in [line.split("<answer>:")]]

    # TODO: some test QA pairs are incomplete! remove incomplete QA pairs!
    raw_data_test = [datum for datum in raw_data_test if "<question>:" in datum["question"]]

    def process_qa_pair(question: str, answer: str) -> dict:
        # remove <answer> prefix
        answer = answer.replace("<answer>: ", "").strip()

        # remove historical data and <question> prefix
        question = question.replace("<question>: ", "")
        current_trajectory_prompt = question.split("There is also historical data:")[0].strip()

        # get instruction
        question_prompt = re.search(
            r"(Given the data, At (.+?), Which POI id will user (\d+) visit\? Note that POI id is an integer in the range from 0 to (\d+).)",
            question,
        ).group(1)
        instruction = current_trajectory_prompt + "\n" + question_prompt

        # get user profile from id
        user_id = re.match(r"The following data is a trajectory of user (\d+):", current_trajectory_prompt).group(1)
        user_profile = get_user_profile(user_id, profiles_path)

        # create system prompt
        system_prompt = create_system_prompt(user_profile, user_id)

        # create llama SFT prompt
        llama_prompt = LLAMA_PROMPT.format(profile=system_prompt, instruction=instruction, answer=answer)

        return {"system_prompt": system_prompt, "inputs": instruction, "targets": answer, "llama_prompt": llama_prompt}

    train_data = [process_qa_pair(**datum) for datum in tqdm(raw_data_train)]
    test_data = [process_qa_pair(**datum) for datum in tqdm(raw_data_test)]

    train_ds, test_ds = Dataset.from_pandas(pd.DataFrame(train_data)), Dataset.from_pandas(pd.DataFrame(test_data))
    dataset = DatasetDict({"train": train_ds, "test": test_ds})
    dataset.push_to_hub(args.dataset_id, private=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
