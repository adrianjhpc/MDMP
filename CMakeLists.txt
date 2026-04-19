cmake_minimum_required(VERSION 3.13)
project(MDMPImplementation LANGUAGES CXX C)

# ==========================================
# NEW: Enable the global CTest test runner
# ==========================================
include(CTest)
enable_testing()

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(MDMP_ENABLE_FORTRAN "Build the Fortran interface (requires a Fortran compiler)" ON)

if(MDMP_ENABLE_FORTRAN)
    enable_language(Fortran)
    message(STATUS "Fortran support enabled. Using compiler: ${CMAKE_Fortran_COMPILER}")
endif()

# Find required packages
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
find_package(MPI REQUIRED C)

# Check if we have the required LLVM components
if(NOT LLVM_ENABLE_PROJECTS)
    set(LLVM_ENABLE_PROJECTS "clang;lld")
endif()

if(NOT LLVM_ENABLE_RTTI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
    message(STATUS "LLVM RTTI is disabled. Disabling RTTI for MDMP.")
endif()

# Include LLVM directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})

# Add definitions from LLVM
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(src)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
add_subdirectory(examples)
