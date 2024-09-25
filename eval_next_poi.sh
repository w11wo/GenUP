#!/bin/bash

#$ -j y
#$ -P CRUISE
#$ -e logs/$JOB_ID_$JOB_NAME.err
#$ -o logs/$JOB_ID_$JOB_NAME.out
#$ -l walltime=5:00:00
#$ -l mem=128G
#$ -l jobfs=100G
#$ -l tmpfree=100G
#$ -l ngpus=1
#$ -l gpu_model=H100_NVL

export HF_HOME=/srv/scratch/CRUISE/Wilson/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
export MAMBA_EXE='/import/glass/1/z5601796/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/srv/scratch/CRUISE/Wilson/micromamba';
export PYTHON_EXEC="$MAMBA_EXE run -n base python";
export PIP_EXEC="$MAMBA_EXE run -n base pip";
export ACCELERATE_EXEC="$MAMBA_EXE run -n base accelerate";

dir=/import/glass/1/z5601796/Wilson/NL-Summ-LLM

nvidia-smi

$PIP_EXEC list

cd $dir

base_model="Meta-Llama-3.1-8B"
dataset_name="FourSquare-TKY-POI"

HF_HUB_OFFLINE=1 $ACCELERATE_EXEC launch --num_processes=1 src/eval_next_poi.py \
    --model_checkpoint "$base_model-$dataset_name" \
    --dataset_id "w11wo/$dataset_name" \
    --apply_liger_kernel_to_llama \
    --temperature 0.65 \
    --top_k 50 \
    --top_p 0.92 \
    --typical_p 0.95 \
    --repetition_penalty 1.0