# NL-Summ-LLM

## Usage

### Generate User Profiles

```sh
python src/generate_user_profile.py --dataset nyc
python src/generate_user_profile.py --dataset ca
python src/generate_user_profile.py --dataset tky
```

### Create SFT Dataset

```sh
python src/create_sft_dataset.py --dataset nyc --dataset_id w11wo/FourSquare-NYC-POI
python src/create_sft_dataset.py --dataset ca --dataset_id w11wo/FourSquare-CA-POI
python src/create_sft_dataset.py --dataset tky --dataset_id w11wo/FourSquare-TKY-POI
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

### Next POI Evaluation

```sh
accelerate launch src/eval_next_poi.py \
    --model_checkpoint "w11wo/Llama-2-7b-longlora-32k-merged-FourSquare-NYC-POI" \
    --dataset_id "w11wo/FourSquare-NYC-POI" \
    --apply_liger_kernel_to_llama
```

## Results

| Model             | History | Others |  NYC   |  TKY   |   CA   |
| ----------------- | :-----: | :----: | :----: | :----: | :----: |
| NL-Summ-Llama2-7b |    ×    |   ×    | 0.2554 | 0.1671 | 0.1130 |
| LLM4POI*          |    ×    |   ×    | 0.2356 | 0.1517 | 0.1016 |
| LLM4POI**         |    ✓    |   ×    | 0.3171 | 0.2836 | 0.1683 |
| LLM4POI**         |    ✓    |   ✓    | 0.3372 | 0.3035 | 0.2065 |