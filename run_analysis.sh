base_model=Llama-3.2-1B

for dataset in "FourSquare-NYC-POI" "Gowalla-CA-POI" "FourSquare-TKY-POI" "FourSquare-Moscow-POI" "FourSquare-SaoPaulo-POI"
do
    python src/user_cold_start_analysis.py \
        --model_checkpoint $base_model-$dataset \
        --dataset_id w11wo/$dataset
    
    python src/trajectory_length_analysis.py \
        --model_checkpoint $base_model-$dataset \
        --dataset_id w11wo/$dataset
done