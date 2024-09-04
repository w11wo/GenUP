#!/bin/bash

#PBS -P hn98
#PBS -q gpuvolta
#PBS -l walltime=0:30:00
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=48GB

dir=/home/561/ww5368/ww5368/NL-Summ-LLM

nvidia-smi

which python
which pip
pip list

cd $dir

base_model="Llama-2-7b-longlora-32k-merged"
dataset_name="FourSquare-NYC-POI"

accelerate launch src/eval_next_poi.py \
    --model_checkpoint "$base_model-$dataset_name" \
    --dataset_id "w11wo/$dataset_name"