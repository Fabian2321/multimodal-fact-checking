#!/bin/bash

# Check if model name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model_name>"
    echo "Available models:"
    echo "  all-MiniLM-L6-v2"
    echo "  all-mpnet-base-v2"
    echo "  multi-qa-mpnet-base-dot-v1"
    exit 1
fi

MODEL_NAME=$1

# Set up logging directory
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Get current timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/optimization_${MODEL_NAME}_${TIMESTAMP}.log"

# Function to handle script termination
cleanup() {
    echo "Script terminated. Check $LOG_FILE for details."
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

# Run the optimization script for the specific model
echo "Starting optimization run for $MODEL_NAME at $(date)" | tee -a $LOG_FILE
echo "Logging to $LOG_FILE" | tee -a $LOG_FILE

# Run with reduced memory threshold and longer pause duration
python src/run_optimization_chunk.py \
    --test_queries_path data/test_queries.json \
    --knowledge_base_path data/knowledge_base \
    --output_path data/optimization_results \
    --memory_threshold 0.8 \
    --pause_duration 600 \
    --model_name "$MODEL_NAME" \
    2>&1 | tee -a $LOG_FILE

echo "Optimization completed for $MODEL_NAME at $(date)" | tee -a $LOG_FILE 