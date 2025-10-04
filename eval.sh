#!/bin/bash

# Environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_TORCH_COMPILE_LEVEL=0
export TORCH_COMPILE_DISABLE=1
export VLLM_USE_MODELSCOPE=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1800
export TOKENIZERS_PARALLELISM=false

export INPUT_MAX_LENGTH=1024
export THINKING_MAX_LENGTH=2048
export POOLING_METHOD=mean
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.4

export HF_DATASETS_CACHE="huggingface_cache"
export HF_HOME="huggingface_cache"


export TRAIN_MODEL_NAME=checkpoints/
export TARGET_DIR=merge_models/

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "Starting model merge..."
# Merge model
python scripts/merge_model.py merge --backend fsdp \
    --hf_model_path Qwen/Qwen2.5-1.5B-Instruct \
    --local_dir "$TRAIN_MODEL_NAME" \
    --target_dir "$TARGET_DIR"

# echo "Model merge completed. Starting evaluation..."

# Task configuration by category
# Retrieval (15 tasks)
RETRIEVAL_TASKS="ArguAna CQADupstackRetrieval FiQA2018 NFCorpus SCIDOCS SciFact ClimateFEVER DBPedia NQ FEVER HotpotQA MSMARCO QuoraRetrieval TRECCOVID Touche2020"

# Reranking (4 tasks)
RERANKING_TASKS="AskUbuntuDupQuestions MindSmallReranking SciDocsRR StackOverflowDupQuestions"

# Clustering (11 tasks)
CLUSTERING_TASKS="ArxivClusteringP2P ArxivClusteringS2S BiorxivClusteringP2P BiorxivClusteringS2S MedrxivClusteringP2P MedrxivClusteringS2S RedditClustering RedditClusteringP2P StackExchangeClustering TwentyNewsgroupsClustering"

# Pair Classification (3 tasks)
PAIR_CLASSIFICATION_TASKS="SprintDuplicateQuestions TwitterSemEval2015 TwitterURLCorpus"

# Classification (12 tasks)
CLASSIFICATION_TASKS="AmazonCounterfactualClassification AmazonPolarityClassification AmazonReviewsClassification Banking77Classification EmotionClassification ImdbClassification ToxicConversationsClassification MTOPDomainClassification MassiveIntentClassification MassiveScenarioClassification MTOPIntentClassification TweetSentimentClassification"

# STS - Semantic Textual Similarity (10 tasks)
STS_TASKS="BIOSSES SICK-R STS12 STS13 STS14 STS15 STS16 STS17 STSBenchmark STS22"

# Summarization (1 task)
SUMMARIZATION_TASKS="SummEval"

# All tasks combined (excluding retrieval for now)
ALL_TASKS="$STS_TASKS $SUMMARIZATION_TASKS $RETRIEVAL_TASKS $RERANKING_TASKS $CLUSTERING_TASKS $PAIR_CLASSIFICATION_TASKS $CLASSIFICATION_TASKS"

# Subset tasks
SUBSET_TASKS="Banking77Classification EmotionClassification MedrxivClusteringS2S TwitterSemEval2015 AskUbuntuDupQuestions BIOSSES STS17 STSBenchmark SummEval"

# Choose which tasks to evaluate (modify this line as needed)
EVAL_TASKS=$ALL_TASKS # or $SUBSET_TASKS

# Model configuration
MODEL_NAME="GRACE-1.5B"
MODEL_PATH="$TARGET_DIR"


# Initialize timing log file
TIMING_LOG="timing_results_$(date +%Y%m%d_%H%M%S).txt"
echo "Task Timing Report - $(date)" > $TIMING_LOG
echo "=================================" >> $TIMING_LOG

# Overall start time
OVERALL_START=$(date +%s)

echo "Starting evaluation..."

python eval_mteb.py \
    --model_path "$MODEL_PATH" \
    --tasks $EVAL_TASKS \
    --output_dir results \
    --batch_size 16 \
    --input_max_length $INPUT_MAX_LENGTH \
    --thinking_max_length $THINKING_MAX_LENGTH \
    --vllm_tensor_parallel_size $VLLM_TENSOR_PARALLEL_SIZE \
    --vllm_gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --model_name "$MODEL_NAME" \
    --pooling_method $POOLING_METHOD

# Overall end time
OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))
OVERALL_MINUTES=$(echo "scale=2; $OVERALL_DURATION / 60" | bc)

echo "=================================" >> $TIMING_LOG
echo "Total evaluation time: $OVERALL_MINUTES minutes" >> $TIMING_LOG
echo "All tasks: $EVAL_TASKS" >> $TIMING_LOG
echo "=================================" >> $TIMING_LOG

# Print timing report
echo ""
echo "====== TIMING REPORT ======"
cat $TIMING_LOG
echo "==========================="

echo "Evaluation completed! All tasks processed with single model load."
