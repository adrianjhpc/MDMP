# ==============================================================================
# MDMP Examples CMake Configuration
# ==============================================================================

# Find standard MPI for the baseline comparison programs
find_package(MPI REQUIRED)

# Define the paths to both of our custom compiler harnesses
set(MDMP_HARNESS_C "${CMAKE_SOURCE_DIR}/harness/mdmpc")
set(MDMP_HARNESS_CXX "${CMAKE_SOURCE_DIR}/harness/mdmpc++")

# ------------------------------------------------------------------------------
# Helper Function: add_comparison_example
# Generates both the standard MPI and the optimized MDMP targets automatically.
# Usage: add_comparison_example(<base_name> <extension> [extra_link_flags...])
# ------------------------------------------------------------------------------
function(add_comparison_example EXAMPLE_NAME EXT)
    # 1. Baseline: Standard MPI Version
    add_executable(${EXAMPLE_NAME}_mpi ${EXAMPLE_NAME}_mpi.${EXT})
    target_include_directories(${EXAMPLE_NAME}_mpi PRIVATE ${MPI_C_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})
    
    # ${ARGN} grabs any optional extra arguments passed to the function (like -lm)
    target_link_libraries(${EXAMPLE_NAME}_mpi PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${ARGN})

    # 2. Determine the correct MDMP harness based on the file extension
    if("${EXT}" STREQUAL "c")
        set(CURRENT_HARNESS ${MDMP_HARNESS_C})
    else()
        set(CURRENT_HARNESS ${MDMP_HARNESS_CXX})
    endif()

    # 3. Optimized: MDMP Version (Compiled via our dynamically chosen LLVM Harness)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp
        COMMAND ${CURRENT_HARNESS} ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}_mdmp.${EXT}
                -I${CMAKE_SOURCE_DIR}/src
                -I/usr/include/mpich-x86_64
                /usr/lib64/mpich/lib/libmpi.so
                ${ARGN} # Inject extra link flags here too
                -o ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}_mdmp.${EXT} mdmp_pragma_pass mdmp_runtime
        COMMENT "Compiling ${EXAMPLE_NAME}_mdmp using ${CURRENT_HARNESS}..."
    )

    # Create a target so `make` knows to trigger the custom command above
    add_custom_target(mdmp_example_${EXAMPLE_NAME} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp)
endfunction()

# ==============================================================================
# Register Example Programs Below
# ==============================================================================

# Heat equation (Pure C, requires the math library: -lm)
add_comparison_example(heat c -lm)

# Circuit satisfiability problem (Standard C++)
add_comparison_example(satisfy cpp)

# Parallel Solution of the 1D Wave Equation
add_comparison_example(wave cpp)
