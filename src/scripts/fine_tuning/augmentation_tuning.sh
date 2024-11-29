#!/bin/bash

model_name="resnet18"
objective="regression"
augmentation_space="weak weak weak weak weak moderate moderate moderate moderate moderate"
batch_size=16
train_chunking_strategy="random"
eval_chunking_strategy="truncate"
ram_optimized_mode=False
max_epochs=30

for augmentation in $augmentation_space; do
    python3 $PROJECT_DIR/src/scripts/fine_tune_cnn.py \
      --model_name=${model_name} \
      --objective=${objective} \
      --train_chunking_strategy=${train_chunking_strategy} \
      --eval_chunking_strategy=${eval_chunking_strategy} \
      --batch_size=${batch_size} \
      --ram_optimized_mode=${ram_optimized_mode} \
      --max_epochs=${max_epochs} \
      --augmentation=${augmentation}
done
