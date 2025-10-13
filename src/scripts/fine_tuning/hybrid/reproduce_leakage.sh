#!/bin/bash

data_leakage=True

# training params
cnn_model_name="resnet18"
encoder_model_name_or_path="roberta-large"
pretrained=True
objective="regression"
augmentation="weak"
batch_size=16
lr="1e-5"
ram_optimized_mode=False
max_epochs=10
enable_checkpointing=False

# cross-validation params
n_folds=5
n_repetitions=5

python3 $PROJECT_DIR/src/scripts/fine_tuning/hybrid/repeated_fine_tune_hybrid.py \
  --cnn_model_name=${cnn_model_name} \
  --encoder_model_name_or_path=${encoder_model_name_or_path} \
  --pretrained=${pretrained} \
  --objective=${objective} \
  --data_leakage=${data_leakage} \
  --batch_size=${batch_size} \
  --ram_optimized_mode=${ram_optimized_mode} \
  --max_epochs=${max_epochs} \
  --augmentation=${augmentation} \
  --lr=${lr} \
  --enable_checkpointing=${enable_checkpointing} \
  --n_folds=${n_folds} \
  --n_repetitions=${n_repetitions}
