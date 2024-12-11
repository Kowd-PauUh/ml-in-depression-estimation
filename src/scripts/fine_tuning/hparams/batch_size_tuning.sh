#!/bin/bash

# training params
model_name="resnet18"
objective="regression"
augmentation="None"
batch_size_space="4 8 16 32 48 64"
lr="1e-5"
train_chunking_strategy="transformer"
eval_chunking_strategy="transformer"
ram_optimized_mode=False
max_epochs=30

# cross-validation params
n_folds=5
n_repetitions=2

for batch_size in $batch_size_space; do
    python3 $PROJECT_DIR/src/scripts/fine_tuning/repeated_fine_tune_cnn.py \
      --model_name=${model_name} \
      --objective=${objective} \
      --train_chunking_strategy=${train_chunking_strategy} \
      --eval_chunking_strategy=${eval_chunking_strategy} \
      --batch_size=${batch_size} \
      --ram_optimized_mode=${ram_optimized_mode} \
      --max_epochs=${max_epochs} \
      --augmentation=${augmentation} \
      --lr=${lr} \
      --n_folds=${n_folds} \
      --n_repetitions=${n_repetitions} \
      --tags="{\"tuned_hparam\":\"batch_size\"}"
done
