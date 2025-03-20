#!/usr/bin/env bash
# Usage:
#   ./read_dataset_labels.sh <label_dataset_input> <label_tensor_output> <label_index>
# Example:
#   ./read_dataset_labels.sh mnist-datasets/train-labels.idx1-ubyte label_out.txt 0

if [ $# -ne 3 ]; then
  echo "Usage: $0 <label_dataset_input> <label_tensor_output> <label_index>"
  exit 1
fi

LABEL_DATASET_INPUT="$1"
LABEL_TENSOR_OUTPUT="$2"
LABEL_INDEX="$3"

echo "This script should read a dataset label into a tensor and pretty-print it into a text file..."
echo "Reading label data from $LABEL_DATASET_INPUT..."
echo "Saving output to $LABEL_TENSOR_OUTPUT"
echo "Label index: $LABEL_INDEX"

# Make sure that mnist_io has been built and the path is correct.
./build/mnist_io "$LABEL_DATASET_INPUT" "$LABEL_TENSOR_OUTPUT" "$LABEL_INDEX"
