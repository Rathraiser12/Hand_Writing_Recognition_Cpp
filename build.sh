#!/bin/bash
echo "This script should build your project now..."

mkdir -p build
cd build

# Generate Makefiles with Release mode
cmake -DCMAKE_BUILD_TYPE=Release ..

# Create executable
make
