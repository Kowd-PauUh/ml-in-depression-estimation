#!/bin/bash

model_name="resnet18"
objective="regression"
batch_size_space="16 32 48 64 80 96"
train_chunking_strategy="random"
eval_chunking_strategy="truncate"
ram_optimized_mode=False
max_epochs=30

for batch_size in $batch_size_space; do
    python3 $PROJECT_DIR/src/scripts/fine_tune_cnn.py \
      --model_name=${model_name} \
      --objective=${objective} \
      --train_chunking_strategy=${train_chunking_strategy} \
      --eval_chunking_strategy=${eval_chunking_strategy} \
      --batch_size=${batch_size} \
      --ram_optimized_mode=${ram_optimized_mode} \
      --max_epochs=${max_epochs}
done
