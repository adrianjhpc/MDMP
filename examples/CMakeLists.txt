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
add_executable(mdmp_example
    example_usage.cpp
)

# Link against MDMP pass and MPI
target_link_libraries(mdmp_example 
    mdmp_pragma_pass
    ${MPI_LIBRARIES}
)

# Set compile options for example
target_compile_options(mdmp_example PRIVATE
    -Wall
    -Wextra
    -pedantic
    ${MPI_CXX_COMPILE_FLAGS}
)

install(TARGETS mdmp_example
    RUNTIME DESTINATION bin
    COMPONENT MDMPExample
)

# Create a simple test
enable_testing()
add_test(NAME mdmp_example_test
    COMMAND mdmp_example
)

