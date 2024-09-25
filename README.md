# NL-Summ-LLM

## Usage

### Download Raw Dataset

```sh
git clone https://huggingface.co/datasets/w11wo/LLM4POI
```

### Generate User Profiles

```sh
python src/generate_user_profile.py --dataset nyc --dataset_id w11wo/FourSquare-NYC-User-Profiles
python src/generate_user_profile.py --dataset ca --dataset_id w11wo/FourSquare-CA-User-Profiles
python src/generate_user_profile.py --dataset tky --dataset_id w11wo/FourSquare-TKY-User-Profiles
```

### Generate POI Reasoning

```sh
python src/generate_poi_reasoning.py --dataset_id w11wo/FourSquare-NYC-POI --dataset nyc
```

### Create SFT Dataset

```sh
python src/create_sft_dataset.py --dataset nyc --dataset_id w11wo/FourSquare-NYC-POI
python src/create_sft_dataset.py --dataset ca --dataset_id w11wo/FourSquare-CA-POI
python src/create_sft_dataset.py --dataset tky --dataset_id w11wo/FourSquare-TKY-POI
python src/create_sft_dataset.py --dataset nyc --dataset_id w11wo/FourSquare-NYC-POI-CoT --use_cot
```

### SFT Training with QLoRA and FSDP

#### Example: Llama-2-7B-LongLoRA-32k on FourSquare-NYC-POI

Train the model using QLoRA and FSDP on Llama-2-7B-LongLoRA-32k with the FourSquare-NYC-POI dataset. Runs on 2 x V100 GPUs.

```sh
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 src/train_sft_qlora_fsdp.py \
    --model_checkpoint "w11wo/Llama-2-7b-longlora-32k-merged" \
    --max_length 16384 \
    --batch_size 1 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --num_epochs 3 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_llama \
    --dataset_id "w11wo/FourSquare-NYC-POI"
```

#### Example: Llama-2-7B-LongLoRA-32k on FourSquare-CA-POI

Train the model using QLoRA and FSDP on Llama-2-7B-LongLoRA-32k with the FourSquare-CA-POI dataset. Runs on 2 x V100 GPUs.

```sh
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 src/train_sft_qlora_fsdp.py \
    --model_checkpoint "w11wo/Llama-2-7b-longlora-32k-merged" \
    --max_length 16384 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --num_epochs 3 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_llama \
    --dataset_id "w11wo/FourSquare-CA-POI"
```

#### Example: Llama-2-7B-LongLoRA-32k on FourSquare-TKY-POI

Train the model using QLoRA and FSDP on Llama-2-7B-LongLoRA-32k with the FourSquare-TKY-POI dataset. Runs on 2 x V100 GPUs.

```sh
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 src/train_sft_qlora_fsdp.py \
    --model_checkpoint "w11wo/Llama-2-7b-longlora-32k-merged" \
    --max_length 16384 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --num_epochs 3 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_llama \
    --dataset_id "w11wo/FourSquare-TKY-POI"
```

#### Example: Llama-3.1-8B on FourSquare-NYC-POI

Train the model using QLoRA and FSDP on Llama-3.1-8B with the FourSquare-NYC-POI dataset. Runs on 2 x H100 GPUs.

```sh
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 src/train_sft_qlora_fsdp.py \
    --model_checkpoint "meta-llama/Meta-Llama-3.1-8B" \
    --max_length 16384 \
    --batch_size 1 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --num_epochs 3 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_llama \
    --dataset_id "w11wo/FourSquare-NYC-POI"
```

#### Example: Llama-3.1-8B on FourSquare-CA-POI

Train the model using QLoRA and FSDP on Llama-3.1-8B with the FourSquare-CA-POI dataset. Runs on 2 x H100 GPUs.

```sh
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 src/train_sft_qlora_fsdp.py \
    --model_checkpoint "meta-llama/Meta-Llama-3.1-8B" \
    --max_length 16384 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --num_epochs 3 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_llama \
    --dataset_id "w11wo/FourSquare-CA-POI"
```

#### Example: Llama-3.1-8B on FourSquare-TKY-POI

Train the model using QLoRA and FSDP on Llama-3.1-8B with the FourSquare-TKY-POI dataset. Runs on 2 x H100 GPUs.

```sh
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 src/train_sft_qlora_fsdp.py \
    --model_checkpoint "meta-llama/Meta-Llama-3.1-8B" \
    --max_length 16384 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --num_epochs 3 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_llama \
    --dataset_id "w11wo/FourSquare-TKY-POI"
```

## Evaluation & Analyses

### Next POI Evaluation

```sh
accelerate launch src/eval_next_poi.py \
    --model_checkpoint "w11wo/Llama-2-7b-longlora-32k-merged-FourSquare-NYC-POI" \
    --dataset_id "w11wo/FourSquare-NYC-POI" \
    --apply_liger_kernel_to_llama
```

```sh
accelerate launch src/eval_next_poi.py \
    --model_checkpoint "w11wo/Meta-Llama-3.1-8B-FourSquare-TKY-POI" \
    --dataset_id "w11wo/FourSquare-TKY-POI" \
    --apply_liger_kernel_to_llama \
    --temperature 0.65 \
    --top_k 50 \
    --top_p 0.92 \
    --typical_p 0.95 \
    --repetition_penalty 1.0
```

### User Cold-start Analysis

```sh
python src/user_cold_start_analysis.py \
    --model_checkpoint w11wo/Llama-2-7b-longlora-32k-merged-FourSquare-NYC-POI \
    --dataset_id w11wo/FourSquare-NYC-POI
```

### Trajectory Length Analysis

```sh
python src/trajectory_length_analysis.py \
    --model_checkpoint w11wo/Llama-2-7b-longlora-32k-merged-FourSquare-NYC-POI \
    --dataset_id w11wo/FourSquare-NYC-POI
```

## POI Prediction Results

| Model               | History | Others |  NYC   |  TKY   |   CA   |
| ------------------- | :-----: | :----: | :----: | :----: | :----: |
| NL-Summ-Llama2-7b   |    ×    |   ×    | 0.2554 | 0.1671 | 0.1130 |
| NL-Summ-Llama3.1-8b |    ×    |   ×    | 0.2582 | 0.2127 | 0.1339 |
| LLM4POI*            |    ×    |   ×    | 0.2356 | 0.1517 | 0.1016 |
| LLM4POI**           |    ✓    |   ×    | 0.3171 | 0.2836 | 0.1683 |
| LLM4POI**           |    ✓    |   ✓    | 0.3372 | 0.3035 | 0.2065 |

## User Cold-start Analysis Results

| User Groups | Model             |  NYC   |  TKY   |   CA   |
| ----------- | ----------------- | :----: | :----: | :----: |
| Inactive    | NL-Summ-Llama2-7b | 0.2042 | 0.1264 | 0.1247 |
| Normal      | NL-Summ-Llama2-7b | 0.2720 | 0.1413 | 0.0967 |
| Very Active | NL-Summ-Llama2-7b | 0.2702 | 0.2018 | 0.1137 |

## Trajectory Length Analysis Results

| Trajectory Length | Model             |  NYC   |  TKY   |   CA   |
| ----------------- | ----------------- | :----: | :----: | :----: |
| Short             | NL-Summ-Llama2-7b | 0.1963 | 0.1117 | 0.0640 |
| Middle            | NL-Summ-Llama2-7b | 0.2684 | 0.1590 | 0.1190 |
| Long              | NL-Summ-Llama2-7b | 0.3117 | 0.2297 | 0.1666 |

## Generalization to Other Datasets

| Trained on |  NYC   |  TKY   |   CA   |
| ---------: | :----: | :----: | :----: |
|        NYC | 0.2554 | 0.1438 | 0.0920 |
|        TKY | 0.2484 | 0.1671 | 0.0996 |
|         CA | 0.2281 | 0.1446 | 0.1130 |