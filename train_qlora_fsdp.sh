#!/bin/bash

#PBS -P hn98
#PBS -q gpuvolta
#PBS -l walltime=12:00:00
#PBS -l ngpus=2
#PBS -l ncpus=24
#PBS -l mem=96GB

dir=/home/561/ww5368/ww5368/NL-Summ-LLM

nvidia-smi

which python
which pip
pip list

cd $dir

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