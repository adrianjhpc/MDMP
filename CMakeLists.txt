cmake_minimum_required(VERSION 3.13)
project(MDMPPragmaImplementation LANGUAGES CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find required packages
find_package(LLVM REQUIRED CONFIG)
find_package(MPI REQUIRED C)

# Check if we have the required LLVM components
if(NOT LLVM_ENABLE_PROJECTS)
    set(LLVM_ENABLE_PROJECTS "clang;lld")
endif()

# Include LLVM directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})

# Add definitions from LLVM
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
