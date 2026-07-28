#!/bin/bash
#SBATCH --job-name=dummy_optuna
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
study_name="dummy"

# Activate environment "${envPath}"

#########################################################

python3 -u "${projectPath}/main.py" \
    --model_path "answerdotai/ModernBERT-base" \
    --tokenizer_path "answerdotai/ModernBERT-base" \
    --do_train \
    --train_path "${projectPath}/data/${trainFile}" \
    --do_dev \
    --dev_path "${projectPath}/data/${devFile}" \
    --no-do_test \
    --seed 42 \
    --is_optuna_trial \
    --study_name "${study_name}_0" \
    --storage_name "sqlite:///${projectPath}/results/${study_name}_dummy.db" \
    --n_trials 2 \
    --max_length 256 \
