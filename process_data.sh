#!/bin/bash

set -e  # Exit on any error

echo "Processing supervised data..."

python process_data.py \
    --input_file echo_data_total.jsonl \
    --local_dir data/supervised \
    --mode supervised \
    --test_ratio 0.01

sleep 3

python offline_filter_data.py \
    --train_parquet data/supervised/train.parquet \
    --val_parquet   data/supervised/test.parquet \
    --out_dir       data/supervised_filtered_overlong/ \
    --tokenizer_path Qwen/Qwen2.5-1.5B-Instruct \
    --max_len 1024

# echo "Processing unsupervised data..."

python process_data.py \
    --input_file wiki1m_for_simcse.txt \
    --local_dir data/unsupervised \
    --mode unsupervised \
    --test_ratio 0.01

sleep 3

python offline_filter_data.py \
    --train_parquet data/unsupervised/train.parquet \
    --val_parquet   data/unsupervised/test.parquet \
    --out_dir       data/unsupervised_filtered_overlong/ \
    --tokenizer_path Qwen/Qwen2.5-1.5B-Instruct \
    --max_len 1024
