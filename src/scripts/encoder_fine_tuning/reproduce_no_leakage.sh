#!/bin/bash

data_leakage=False

# training params
model_name_or_path="roberta-large"
objective="regression"
batch_size=16
lr="3e-5"
max_epochs=4
lr_patience=10
enable_checkpointing=False

# cross-validation params
n_folds=5
n_repetitions=5

python3 $PROJECT_DIR/src/scripts/encoder_fine_tuning/repeated_fine_tune_encoder.py \
  --model_name_or_path=${model_name_or_path} \
  --objective=${objective} \
  --data_leakage=${data_leakage} \
  --batch_size=${batch_size} \
  --max_epochs=${max_epochs} \
  --lr=${lr} \
  --lr_patience=${lr_patience} \
  --enable_checkpointing=${enable_checkpointing} \
  --n_folds=${n_folds} \
  --n_repetitions=${n_repetitions} \
  --tags="{\"reproduce_depresnet\":\"true\"}"
