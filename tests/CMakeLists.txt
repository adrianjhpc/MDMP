# tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.13)

# Create example application
add_executable(mdmp_test test_mdmp.cpp)

# Use target_include_directories with absolute paths to guarantee visibility
target_include_directories(mdmp_test PRIVATE 
    ${CMAKE_SOURCE_DIR}/src 
    ${LLVM_INCLUDE_DIRS}
    ${MPI_INCLUDE_PATH}
)

# Link against the MDMP runtime and MPI
target_link_libraries(mdmp_test PRIVATE
    mdmp_runtime
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
add_test(NAME mdmp_example_test COMMAND mdmp_test)
