

cmake_minimum_required(VERSION 3.10)

project(MNIST_Reader)

# Automatically find and set the C and C++ compilers
find_program(CMAKE_C_COMPILER NAMES gcc)
find_program(CMAKE_CXX_COMPILER NAMES g++)

if(NOT CMAKE_C_COMPILER)
  message(FATAL_ERROR "C compiler not found. Please install GCC.")
endif()

if(NOT CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "C++ compiler not found. Please install G++.")
endif()



set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD_REQUIRED ON)



# Add source files

add_executable(mnist_reader

  src/main.cpp

  src/mnist_reader.cpp

)



# Add include directories (to find header files)

target_include_directories(mnist_reader PRIVATE

  ${CMAKE_SOURCE_DIR}/src

)
