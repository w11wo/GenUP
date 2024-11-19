base_model="Meta-Llama-3.1-8B"
dataset_name="FourSquare-TKY-POI"

accelerate launch --num_processes=1 src/eval_next_poi.py \
    --model_checkpoint "$base_model-$dataset_name" \
    --dataset_id "w11wo/$dataset_name" \
    --apply_liger_kernel_to_llama \
    --temperature 0.65 \
    --top_k 50 \
    --top_p 0.92 \
    --typical_p 0.95 \
    --repetition_penalty 1.0