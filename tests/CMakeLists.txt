# CMakeLists.txt for building with LLVM support
cmake_minimum_required(VERSION 3.13)

# Include LLVM directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})

# Add definitions from LLVM
add_definitions(${LLVM_DEFINITIONS})

# Include MDMP directories
include_directories("../src")

# Create example application
add_executable(mdmp_test
    test_mdmp.cpp
)


# Link against MDMP pass and MPI
target_link_libraries(mdmp_test 
    mdmp_pragma_pass
    ${MPI_LIBRARIES}
)

# Set compile options for test
target_compile_options(mdmp_test PRIVATE
    -Wall
    -Wextra
    -pedantic
    ${MPI_CXX_COMPILE_FLAGS}
)

# Create a simple test
enable_testing()
add_test(NAME mdmp_example_test
    COMMAND mdmp_test
)
