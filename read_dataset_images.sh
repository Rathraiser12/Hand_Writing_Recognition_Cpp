#!/usr/bin/env bash
# Usage:
#   ./read_dataset_images.sh <image_dataset_input> <image_tensor_output> <image_index>

if [ $# -ne 3 ]; then
  echo "Usage: $0 <image_dataset_input> <image_tensor_output> <image_index>"
  exit 1
fi

IMAGE_DATASET_INPUT="$1"
IMAGE_TENSOR_OUTPUT="$2"
IMAGE_INDEX="$3"

echo "This script should read a dataset image into a tensor and pretty-print it into a text file..."
echo "Reading image data from $IMAGE_DATASET_INPUT..."
echo "Saving output to $IMAGE_TENSOR_OUTPUT"
echo "Image index: $IMAGE_INDEX"

# Make sure that mnist_io has been built and the path is correct.
./build/mnist_io "$IMAGE_DATASET_INPUT" "$IMAGE_TENSOR_OUTPUT" "$IMAGE_INDEX"
