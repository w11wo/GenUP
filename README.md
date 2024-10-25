# GenUP: Generative User Profilers as In-Context Learners for Next POI Recommender Systems

## Usage

### Download Raw Dataset

```sh
git clone https://huggingface.co/datasets/w11wo/LLM4POI
```

### Generate User Profiles

```sh
python src/generate_user_profile.py --dataset nyc --dataset_id w11wo/FourSquare-NYC-User-Profiles
python src/generate_user_profile.py --dataset ca --dataset_id w11wo/Gowalla-CA-User-Profiles
python src/generate_user_profile.py --dataset tky --dataset_id w11wo/FourSquare-TKY-User-Profiles
```

### Create SFT Dataset

```sh
python src/create_sft_dataset.py --dataset nyc --dataset_id w11wo/FourSquare-NYC-POI
python src/create_sft_dataset.py --dataset ca --dataset_id w11wo/Gowalla-CA-POI
python src/create_sft_dataset.py --dataset tky --dataset_id w11wo/FourSquare-TKY-POI
```

### SFT Training with QLoRA and FSDP

#### Llama-2-7B-LongLoRA-32k

<details>
  <summary>FourSquare-NYC-POI</summary>

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
</details>

<details>
  <summary>Gowalla-CA-POI</summary>

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
    --dataset_id "w11wo/Gowalla-CA-POI"
```
</details>

<details>
  <summary>FourSquare-TKY-POI</summary>

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
</details>

<details>
  <summary>FourSquare-Moscow-POI</summary>

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
    --dataset_id "w11wo/FourSquare-Moscow-POI"
```
</details>

<details>
  <summary>FourSquare-SaoPaulo-POI</summary>

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
    --dataset_id "w11wo/FourSquare-SaoPaulo-POI"
```
</details>

#### Llama-3.1-8B

<details>
  <summary>FourSquare-NYC-POI</summary>

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
</details>

<details>
  <summary>Gowalla-CA-POI</summary>

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
    --dataset_id "w11wo/Gowalla-CA-POI"
```
</details>

<details>
  <summary>FourSquare-TKY-POI</summary>

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
</details>

<details>
  <summary>FourSquare-Moscow-POI</summary>

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
    --dataset_id "w11wo/FourSquare-Moscow-POI"
```
</details>

<details>
  <summary>FourSquare-SaoPaulo-POI</summary>

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
    --dataset_id "w11wo/FourSquare-SaoPaulo-POI"
```
</details>

#### Llama-3.2-1B

<details>
  <summary>FourSquare-NYC-POI</summary>

```sh
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 src/train_sft_qlora_fsdp.py \
    --model_checkpoint "meta-llama/Meta-Llama-3.2-1B" \
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
</details>

<details>
  <summary>Gowalla-CA-POI</summary>

```sh
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 src/train_sft_qlora_fsdp.py \
    --model_checkpoint "meta-llama/Meta-Llama-3.2-1B" \
    --max_length 16384 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --num_epochs 3 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_llama \
    --dataset_id "w11wo/Gowalla-CA-POI"
```
</details>

<details>
  <summary>FourSquare-TKY-POI</summary>

```sh
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 src/train_sft_qlora_fsdp.py \
    --model_checkpoint "meta-llama/Meta-Llama-3.2-1B" \
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
</details>

<details>
  <summary>FourSquare-Moscow-POI</summary>

```sh
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 src/train_sft_qlora_fsdp.py \
    --model_checkpoint "meta-llama/Meta-Llama-3.2-1B" \
    --max_length 16384 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --num_epochs 3 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_llama \
    --dataset_id "w11wo/FourSquare-Moscow-POI"
```
</details>

<details>
  <summary>FourSquare-SaoPaulo-POI</summary>

```sh
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=2 src/train_sft_qlora_fsdp.py \
    --model_checkpoint "meta-llama/Meta-Llama-3.2-1B" \
    --max_length 16384 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --num_epochs 3 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_llama \
    --dataset_id "w11wo/FourSquare-SaoPaulo-POI"
```
</details>

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

```sh
accelerate launch src/eval_next_poi.py \
    --model_checkpoint "w11wo/Meta-Llama-3.2-1B-FourSquare-TKY-POI" \
    --dataset_id "w11wo/Gowalla-CA-POI" \
    --apply_liger_kernel_to_llama \
    --temperature 0.65 \
    --top_k 50 \
    --top_p 0.92 \
    --typical_p 0.95 \
    --repetition_penalty 1.0
```

```sh
accelerate launch src/eval_next_poi.py \
    --model_checkpoint "w11wo/Llama-3.1-8B-FourSquare-Moscow-POI" \
    --dataset_id "w11wo/FourSquare-Moscow-POI" \
    --apply_liger_kernel_to_llama \
    --temperature 0.6 \
    --top_k 50 \
    --top_p 0.1 \
    --typical_p 0.95 \
    --repetition_penalty 1.0
```

```sh
accelerate launch src/eval_next_poi.py \
    --model_checkpoint "w11wo/Meta-Llama-3.2-1B-FourSquare-SaoPaulo-POI" \
    --dataset_id "w11wo/FourSquare-SaoPaulo-POI" \
    --apply_liger_kernel_to_llama \
    --temperature 0.6 \
    --top_k 50 \
    --top_p 0.1 \
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

| Model                                           | History | Other Users |    NYC     |    TKY     |     CA     |
| ----------------------------------------------- | :-----: | :---------: | :--------: | :--------: | :--------: |
| *Without historical and intra-user social data* |
| LLM4POI*                                        |    ×    |      ×      |   0.2356   |   0.1517   |   0.1016   |
| GenUP-Llama2-7b                                 |    ×    |      ×      |   0.2575   |   0.1699   |   0.1094   |
| GenUP-Llama3.1-8b                               |    ×    |      ×      | **0.2582** | **0.2127** | **0.1339** |
| GenUP-Llama3.2-1b                               |    ×    |      ×      |   0.2484   |   0.1851   |   0.1267   |
| *With historical and intra-user social data*    |
| GETNext                                         |    ✓    |      ✓      |   0.2435   |   0.2254   |   0.1357   |
| STHGCN                                          |    ✓    |      ✓      |   0.2734   |   0.2950   |   0.1730   |
| LLM4POI                                         |    ✓    |      ✓      | **0.3372** | **0.3035** | **0.2065** |

| Model                                    | Moscow | Sao Paulo |
| ---------------------------------------- | :----: | :-------: |
| *Supervised fine-tuning*                 |
| LLM4POI*                                 | 0.146  |   0.166   |
| GenUP-Llama2-7b                          | 0.159  |   0.175   |
| GenUP-Llama3.1-8b                        | 0.163  |   0.178   |
| GenUP-Llama3.2-1b                        | 0.161  |   0.175   |
| *In-context learning*                    |
| LLM-Mob                                  | 0.080  |   0.140   |
| LLM-ZS                                   | 0.120  |   0.165   |
| *LLM agents and with external knowledge* |
| AgentMove                                | 0.160  |   0.230   |

## User Cold-start Analysis Results

| User Groups | Model             |  NYC   |  TKY   |   CA   | Moscow | Sao Paulo |
| ----------- | ----------------- | :----: | :----: | :----: | :----: | :-------: |
| Inactive    | GenUP-Llama2-7b   | 0.2105 | 0.1306 | 0.1091 | 0.1227 |  0.1366   |
| Normal      | GenUP-Llama2-7b   | 0.2591 | 0.1394 | 0.1089 | 0.1410 |  0.1504   |
| Very Active | GenUP-Llama2-7b   | 0.2752 | 0.2063 | 0.1096 | 0.1748 |  0.1940   |
| Inactive    | GenUP-Llama3.1-8b | 0.1826 | 0.1486 | 0.1380 | 0.1180 |  0.1393   |
| Normal      | GenUP-Llama3.1-8b | 0.2554 | 0.1695 | 0.1338 | 0.1464 |  0.1598   |
| Very Active | GenUP-Llama3.1-8b | 0.2884 | 0.2688 | 0.1324 | 0.1808 |  0.1944   |
| Inactive    | GenUP-Llama3.2-1b | 0.1764 | 0.1306 | 0.1316 | 0.1210 |  0.1429   |
| Normal      | GenUP-Llama3.2-1b | 0.2664 | 0.1494 | 0.1223 | 0.1390 |  0.1530   |
| Very Active | GenUP-Llama3.2-1b | 0.2704 | 0.2321 | 0.1263 | 0.1793 |  0.1906   |

## Trajectory Length Analysis Results

| Trajectory Length | Model             |  NYC   |  TKY   |   CA   | Moscow | Sao Paulo |
| ----------------- | ----------------- | :----: | :----: | :----: | :----: | :-------: |
| Short             | GenUP-Llama2-7b   | 0.1980 | 0.1138 | 0.0649 | 0.0646 |  0.0706   |
| Middle            | GenUP-Llama2-7b   | 0.2801 | 0.1693 | 0.1154 | 0.1873 |  0.1985   |
| Long              | GenUP-Llama2-7b   | 0.3099 | 0.2264 | 0.1578 | 0.2494 |  0.2739   |
| Short             | GenUP-Llama3.1-8b | 0.2146 | 0.1717 | 0.1070 | 0.0744 |  0.0759   |
| Middle            | GenUP-Llama3.1-8b | 0.2529 | 0.2014 | 0.1367 | 0.1899 |  0.2009   |
| Long              | GenUP-Llama3.1-8b | 0.3064 | 0.2636 | 0.1637 | 0.2490 |  0.2745   |
| Short             | GenUP-Llama3.2-1b | 0.1830 | 0.1423 | 0.1011 | 0.0744 |  0.0762   |
| Middle            | GenUP-Llama3.2-1b | 0.2529 | 0.1730 | 0.1385 | 0.1844 |  0.1913   |
| Long              | GenUP-Llama3.2-1b | 0.3152 | 0.2384 | 0.1500 | 0.2459 |  0.2694   |

## Generalization to Other Datasets

### GenUP-Llama2-7b

| Trained on |  NYC   |  TKY   |   CA   |
| ---------: | :----: | :----: | :----: |
|        NYC | 0.2575 | 0.1438 | 0.0920 |
|        TKY | 0.2484 | 0.1699 | 0.0996 |
|         CA | 0.2281 | 0.1446 | 0.1094 |

### GenUP-Llama3.1-8b

| Trained on |  NYC   |  TKY   |   CA   |
| ---------: | :----: | :----: | :----: |
|        NYC | 0.2127 | 0.1179 | 0.0787 |
|        TKY | 0.1924 | 0.2582 | 0.0848 |
|         CA | 0.1987 | 0.1197 | 0.1339 |

### GenUP-Llama3.2-1b

| Trained on |  NYC   |  TKY   |   CA   |
| ---------: | :----: | :----: | :----: |
|        NYC | 0.2484 | 0.1197 | 0.0787 |
|        TKY | 0.1973 | 0.1851 | 0.0769 |
|         CA | 0.2253 | 0.1236 | 0.1267 |

## Ablation Study

### User Profile Components

| Components                                                 | Model           |  NYC   |
| ---------------------------------------------------------- | --------------- | :----: |
| Profile                                                    | GenUP-Llama2-7b | 0.2568 |
| Profile + Routines & Preferences                           | GenUP-Llama2-7b | 0.2568 |
| Profile + Routines & Preferences + Attributes              | GenUP-Llama2-7b | 0.2575 |
| Profile + Routines & Preferences + Attributes + BFI Traits | GenUP-Llama2-7b | 0.2575 |