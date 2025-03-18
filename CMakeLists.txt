cmake_minimum_required(VERSION 3.10)
project(MNIST_Reader)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If you still want to pick up the C and C++ compilers:
find_program(CMAKE_C_COMPILER NAMES gcc clang cl)
find_program(CMAKE_CXX_COMPILER NAMES g++ clang++ cl)

# ---------------------------
# 1) Executable: mnist_io
#    (Task 1 style: reading/writing MNIST)
# ---------------------------
add_executable(mnist_io
    src/main.cpp          # your old main for MNIST I/O
    src/mnist_reader.cpp  # presumably needed for readDatasetImages, etc.
    # add any other .cpp needed only by the I/O code
)

target_include_directories(mnist_io PRIVATE
    "${CMAKE_SOURCE_DIR}/src"     # if your headers are in src/
    "${CMAKE_SOURCE_DIR}/include" # if you have an include/ folder
)

# ---------------------------
# 2) Executable: nn_trainer
#    (Task 2 style: neural network training/testing)
# ---------------------------
add_executable(nn_trainer
    src/test_train_model.cpp  # new main for training
    # Possibly add other .cpp files that define your neural network code, e.g.:
    # src/fullyconnected.cpp
    # src/softmax.cpp
    # src/relu.cpp
    # src/neuralnetwork.cpp
    # src/optimizers.cpp
    # src/loss.cpp
    # ...
)

target_include_directories(nn_trainer PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/include"
)
