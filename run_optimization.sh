#!/bin/bash

# Set up logging directory
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Get current timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/optimization_$TIMESTAMP.log"

# Function to handle script termination
cleanup() {
    echo "Script terminated. Check $LOG_FILE for details."
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

# Run the optimization script with resource management
echo "Starting optimization run at $(date)" | tee -a $LOG_FILE
echo "Logging to $LOG_FILE" | tee -a $LOG_FILE

# Run with reduced memory threshold and longer pause duration
python src/run_optimization.py \
    --test_queries_path data/test_queries.json \
    --knowledge_base_path data/knowledge_base \
    --output_path data/optimization_results \
    --memory_threshold 0.8 \
    --pause_duration 600 \
    2>&1 | tee -a $LOG_FILE

echo "Optimization completed at $(date)" | tee -a $LOG_FILE 