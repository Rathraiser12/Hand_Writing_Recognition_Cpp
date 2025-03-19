cmake_minimum_required(VERSION 3.10)
project(MNIST_Reader)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optionally, pick up the C and C++ compilers:
find_program(CMAKE_C_COMPILER NAMES gcc clang cl)
find_program(CMAKE_CXX_COMPILER NAMES g++ clang++ cl)

# Append optimization flags for GCC/Clang in Release mode
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

# Find and configure OpenMP if available
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# ---------------------------
# 1) Executable: mnist_io
#    (Task 1 style: reading/writing MNIST)
# ---------------------------
add_executable(mnist_io
    src/main.cpp          # your old main for MNIST I/O
    src/mnist_reader.cpp  # presumably needed for reading image data, etc.
    # add any other .cpp files needed only by the I/O code
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
    src/test_train_model.cpp  # new main for training/testing
)

target_include_directories(nn_trainer PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/include"
)
