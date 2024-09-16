#!/bin/bash

#$ -j y
#$ -P CRUISE
#$ -e logs/$JOB_ID_$JOB_NAME.err
#$ -o logs/$JOB_ID_$JOB_NAME.out
#$ -l walltime=00:30:00
#$ -l mem=48G
#$ -l jobfs=10G
#$ -l tmpfree=12G
#$ -l ngpus=1
#$ -l gpu_model=L40S

export HF_HOME=/srv/scratch/CRUISE/Wilson/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
export MAMBA_EXE='/import/glass/1/z5601796/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/srv/scratch/CRUISE/Wilson/micromamba';
alias python="$MAMBA_EXE run -n base python";
alias pip="$MAMBA_EXE run -n base pip";
alias accelerate="$MAMBA_EXE run -n base accelerate";

dir=/import/glass/1/z5601796/Wilson/NL-Summ-LLM

nvidia-smi

pip list

cd $dir

base_model="Llama-2-7b-longlora-32k-merged"
dataset_name="FourSquare-NYC-POI"

HF_HUB_OFFLINE=1 accelerate launch src/eval_next_poi.py \
    --model_checkpoint "$base_model-$dataset_name" \
    --dataset_id "w11wo/$dataset_name" \
    --profiles_dataset_id "w11wo/FourSquare-NYC-User-Profiles" \
    --profiles_similarity_path "data/nyc/profile_similarities.json" \
    --top_k_similar_profiles 5 \
    --apply_liger_kernel_to_llama