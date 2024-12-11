#!/bin/bash

# training params
model_name="resnet18"
objective="regression"
augmentation="None"
batch_size=16
lr_space="1e-5 1e-4 1e-3"
train_chunking_strategy="transformer"
eval_chunking_strategy="transformer"
ram_optimized_mode=False
max_epochs=30

# cross-validation params
n_folds=5
n_repetitions=2

for lr in $lr_space; do
    python3 $PROJECT_DIR/src/scripts/repeated_fine_tune_cnn.py \
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
      --n_repetitions=${n_repetitions}
done
