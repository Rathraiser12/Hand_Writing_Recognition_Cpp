cmake_minimum_required(VERSION 3.10)
project(MNIST_Reader)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_program(CMAKE_C_COMPILER NAMES gcc clang cl)
find_program(CMAKE_CXX_COMPILER NAMES g++ clang++ cl)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Target for single image/label I/O (used by your read dataset scripts)
# This target now uses src/main.cpp and src/mnist_data_loader.cpp.
add_executable(mnist_io
  src/main.cpp
  src/mnist_data_loader.cpp
)
target_include_directories(mnist_io PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/include"
)

# Target for neural network training/testing using the integrated loader.
add_executable(nn_trainer
  src/test_train_model.cpp
  src/mnist_data_loader.cpp
)
target_include_directories(nn_trainer PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  "${CMAKE_SOURCE_DIR}/include"
)
