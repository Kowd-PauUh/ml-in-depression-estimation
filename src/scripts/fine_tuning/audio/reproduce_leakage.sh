#!/bin/bash

data_leakage=True

# training params
model_name="resnet18"
pretrained=True
objective="regression"
augmentation="weak"
batch_size=16
lr="1e-5"
train_chunking_strategy="transformer"
eval_chunking_strategy="transformer"
ram_optimized_mode=False
max_epochs=10
enable_checkpointing=False

# cross-validation params
n_folds=5
n_repetitions=5

python3 $PROJECT_DIR/src/scripts/fine_tuning/audio/repeated_fine_tune_cnn.py \
  --model_name=${model_name} \
  --pretrained=${pretrained} \
  --objective=${objective} \
  --data_leakage=${data_leakage} \
  --train_chunking_strategy=${train_chunking_strategy} \
  --eval_chunking_strategy=${eval_chunking_strategy} \
  --batch_size=${batch_size} \
  --ram_optimized_mode=${ram_optimized_mode} \
  --max_epochs=${max_epochs} \
  --augmentation=${augmentation} \
  --lr=${lr} \
  --enable_checkpointing=${enable_checkpointing} \
  --n_folds=${n_folds} \
  --n_repetitions=${n_repetitions}
