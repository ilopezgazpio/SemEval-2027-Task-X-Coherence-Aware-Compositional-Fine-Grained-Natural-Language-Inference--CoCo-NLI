#!/bin/bash
#SBATCH --job-name=dummy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1

#########################################################
projectPath=""
envPath=""
trainFile=""
devFile=""
testFile=""

# Activate environment "${envPath}"

#########################################################

python3 -u "${projectPath}/main.py" \
    --model_path "answerdotai/ModernBERT-base" \
    --tokenizer_path "answerdotai/ModernBERT-base" \
    --do_train \
    --train_path "${projectPath}/data/${trainFile}" \
    --do_dev \
    --dev_path "${projectPath}/data/${devFile}" \
    --do_test \
    --test_path "${projectPath}/data/${testFile}" \
    --seed 42 \
    --dropout 0.1 \
    --warmup_pcrt 0.01 \
    --lr 5e-5 \
    --batch_size 64 \
    --wd 1e-5 \
    --num_epochs 3 \
    --no-is_optuna_trial \
    --max_length 256 \
    --results_dir "${projectPath}/results"
    # --save_path "${projectPath}/results/" \
    # --epoch_to_stop 2 \

python3 -u "${projectPath}/main.py" \
    --model_path "answerdotai/ModernBERT-base" \
    --tokenizer_path "answerdotai/ModernBERT-base" \
    --do_train \
    --train_path "${projectPath}/data/${trainFile}" \
    --no-do_dev \
    --do_test \
    --test_path "${projectPath}/data/${testFile}" \
    --seed 42 \
    --dropout 0.1 \
    --warmup_pcrt 0.01 \
    --lr 5e-5 \
    --batch_size 64 \
    --wd 1e-5 \
    --num_epochs 3 \
    --no-is_optuna_trial \
    --max_length 256 \
    --results_dir "${projectPath}/results"
    # --save_path "${projectPath}/results/" \
    # --epoch_to_stop 2 \

python3 -u "${projectPath}/main.py" \
    --model_path "answerdotai/ModernBERT-base" \
    --tokenizer_path "answerdotai/ModernBERT-base" \
    --do_train \
    --train_path "${projectPath}/data/${trainFile}" \
    --no-do_dev \
    --no-do_test \
    --seed 42 \
    --dropout 0.1 \
    --warmup_pcrt 0.01 \
    --lr 5e-5 \
    --batch_size 64 \
    --wd 1e-5 \
    --num_epochs 3 \
    --no-is_optuna_trial \
    --max_length 256 \
    --results_dir "${projectPath}/results"
    # --save_path "${projectPath}/results/" \
    # --epoch_to_stop 2 \
