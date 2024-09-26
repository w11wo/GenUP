#!/bin/bash

#$ -j y
#$ -P CRUISE
#$ -e logs/$JOB_ID_$JOB_NAME.err
#$ -o logs/$JOB_ID_$JOB_NAME.out
#$ -l walltime=20:00:00
#$ -l mem=256G
#$ -l jobfs=150G
#$ -l tmpfree=150G
#$ -l ngpus=2
#$ -l gpu_model=H100_NVL

export HF_HOME=/srv/scratch/CRUISE/Wilson/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
export MAMBA_EXE='/import/glass/1/z5601796/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/srv/scratch/CRUISE/Wilson/micromamba';
export PIP_EXEC="$MAMBA_EXE run -n base pip";
export ACCELERATE_EXEC="$MAMBA_EXE run -n base accelerate";
export TORCHRUN_EXEC="$MAMBA_EXE run -n base torchrun";

dir=/import/glass/1/z5601796/Wilson/NL-Summ-LLM

nvidia-smi

$PIP_EXEC list

cd $dir

HF_HUB_OFFLINE=1 ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 $TORCHRUN_EXEC --nproc_per_node=2 src/train_sft_qlora_fsdp.py \
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