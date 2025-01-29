

#!/bin/bash



# Create build directory if it doesn't exist

mkdir -p build



# Go to build directory

cd build



# Run cmake

cmake ..



# Build the project

make



# Return to original directory

cd ..


