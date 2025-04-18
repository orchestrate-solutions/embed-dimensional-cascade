#!/bin/bash
# Run validation for dimensional cascade
# This script runs validation tests comparing our dimensional cascade to simple truncation

# Check if embeddings file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <embeddings_file> [models_dir] [output_dir]"
    echo "Example: $0 data/embeddings/arctic_768.npy cascade_models validation_results"
    exit 1
fi

# Set variables
EMBEDDINGS_FILE=$1
MODELS_DIR=${2:-"cascade_models"}
OUTPUT_DIR=${3:-"validation_results"}

# Define k values for recall calculation
K_VALUES="1,5,10,50,100"
NUM_SAMPLES=1000

echo "Starting validation..."
echo "Embeddings file: $EMBEDDINGS_FILE"
echo "Models directory: $MODELS_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run validation
python validate_cascade.py \
    --embeddings "$EMBEDDINGS_FILE" \
    --models-dir "$MODELS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --k-values "$K_VALUES" \
    --num-samples "$NUM_SAMPLES"

# Check if validation was successful
if [ $? -eq 0 ]; then
    echo "Validation completed successfully!"
    echo "Results saved to $OUTPUT_DIR"
    echo "Progress report: $OUTPUT_DIR/progress_report.md"
    echo "Visualization plots: $OUTPUT_DIR/plots/"
else
    echo "Validation failed!"
    exit 1
fi 