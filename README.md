# NL-Summ-LLM

## Generate User Profiles

```sh
python src/generate_user_profile.py --dataset nyc
python src/generate_user_profile.py --dataset ca
```

## Create SFT Dataset

```sh
python src/create_sft_dataset.py --dataset nyc --dataset_id w11wo/FourSquare-NYC-POI
```

## SFT Training with QLoRA and FSDP

### Example: Llama-2-7B-LongLoRA-32k

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
    --dataset_id "w11wo/FourSquare-NYC-POI"
```

### Example: Llama-3.1-8B

```sh
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 src/train_sft_qlora_fsdp.py \
    --model_checkpoint "meta-llama/Meta-Llama-3.1-8B" \
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