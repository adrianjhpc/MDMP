# CMakeLists.txt for building with LLVM support
cmake_minimum_required(VERSION 3.13)

# Include LLVM directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})

# Add definitions from LLVM
add_definitions(${LLVM_DEFINITIONS})

# Create the MDMP pragma pass library
add_library(mdmp_pragma_pass SHARED
    mdmp_pragma_pass.cpp
)

# Link against LLVM libraries
target_link_libraries(mdmp_pragma_pass 
    ${LLVM_LIBRARIES}
    ${MPI_LIBRARIES}
)

# Set library properties
set_target_properties(mdmp_pragma_pass PROPERTIES
    PREFIX ""
    OUTPUT_NAME "MDMPPragma"
)

# Set compile options for the pass
target_compile_options(mdmp_pragma_pass PRIVATE
    ${LLVM_CXXFLAGS}
    -Wall
    -Wextra
    -pedantic
)

# Install targets
install(TARGETS mdmp_pragma_pass
    LIBRARY DESTINATION lib
    COMPONENT mdmp_pragma_pass
)

# Install headers
install(FILES 
    mdmp_runtime.h
    mdmp_runtime.h
    DESTINATION include
    COMPONENT Headers
)

# Print configuration info
message(STATUS "LLVM Version: ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM Include dirs: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM Libraries: ${LLVM_LIBRARIES}")
message(STATUS "MPI Include dirs: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI Libraries: ${MPI_LIBRARIES}")

# Add custom targets for building documentation
add_custom_target(doc
    COMMAND doxygen Doxyfile
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building documentation"
)

# Add custom target for running tests
add_custom_target(tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    COMMENT "Running all tests"
)
