import re
import json

import torch
from argparse import ArgumentParser
from tqdm.auto import tqdm
from accelerate import Accelerator
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftConfig, PeftModel
from liger_kernel.transformers import apply_liger_kernel_to_llama


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--apply_liger_kernel_to_llama", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset(args.dataset_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # NOTE: we've formatted the prompt to include the <s> token at the beginning of the prompt
    if tokenizer.add_bos_token:
        tokenizer.add_bos_token = False

    torch_dtype = torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

    device_index = Accelerator().process_index

    peft_config = PeftConfig.from_pretrained(args.model_checkpoint)

    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        attn_implementation="sdpa",  # alternatively use "flash_attention_2"
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map={"": device_index},
    )

    if args.apply_liger_kernel_to_llama:
        apply_liger_kernel_to_llama()

    model = PeftModel.from_pretrained(model, args.model_checkpoint)
    model.eval()

    cot_generation_config = GenerationConfig(
        max_new_tokens=600,
        min_new_tokens=None,
        do_sample=True,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.6,
        top_k=40,
        top_p=0.1,
        typical_p=1.0,
        repetition_penalty=1.176,
        num_return_sequences=1,
        stop_strings="</think>",
    )

    generation_config = GenerationConfig(
        max_new_tokens=5,
        min_new_tokens=None,
        do_sample=True,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.6,
        top_k=40,
        top_p=0.1,
        typical_p=1.0,
        repetition_penalty=1.176,
        num_return_sequences=1,
    )

    predictions, targets = [], []

    lines = dataset["test"]["llama_prompt"]
    for line in tqdm(lines):
        # split prompt with target POI
        # ex. prompt = <s>[INST] <<SYS>> ... [/INST] At 2013-01-03 10:13:27, user 1 will visit POI id
        # ex. target = 123. </s>
        instruction, closing_instruct_tag, _, target = re.split(r"(\[/INST\])(.+?)", line)
        output_prompt, target, _ = re.split(r"(\d+\.\s</s>)", target)
        target = re.sub(r"[^0-9]", "", target)  # remove non-numeric tokens

        prompt = instruction + closing_instruct_tag + "<think>"
        prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**prompt_input_ids, generation_config=cot_generation_config, tokenizer=tokenizer)

        cot_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # get content inside <think> tags
        cot_output = re.findall(r"<think>(.*?)</think>", cot_output, re.DOTALL)[0]

        prompt = prompt + cot_output + "</think>" + "<output>" + output_prompt

        prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_token_length = prompt_input_ids.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(**prompt_input_ids, generation_config=generation_config)

        prediction = tokenizer.decode(outputs[0, prompt_token_length:], skip_special_tokens=True)
        prediction = re.sub(r"[^0-9]", "", prediction)  # remove non-numeric tokens

        predictions.append(prediction)
        targets.append(target)

    accuracy = accuracy_score(targets, predictions)

    model_id = args.model_checkpoint.split("/")[-1]
    dataset_id = args.dataset_id.split("/")[-1]

    result = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "accuracy": accuracy,
        "predictions": predictions,
        "targets": targets,
    }

    with open(f"results/{model_id}-{dataset_id}.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
