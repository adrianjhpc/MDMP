# ==============================================================================
# MDMP Examples CMake Configuration
# ==============================================================================

find_package(MPI REQUIRED)

set(MDMP_HARNESS_C "${CMAKE_SOURCE_DIR}/harness/mdmpc")
set(MDMP_HARNESS_CXX "${CMAKE_SOURCE_DIR}/harness/mdmpc++")

# Default ranks if a specific example doesn't override it
set(BENCHMARK_RANKS 4)

add_custom_target(run_examples
    COMMENT "Executing all MDMP baseline vs optimized benchmarks..."
)

# ------------------------------------------------------------------------------
# Helper Function: add_comparison_example
# Usage: add_comparison_example(<name> <ext> [RANKS <n>] [LINK_LIBS <libs...>] [RUN_ARGS <args...>])
# ------------------------------------------------------------------------------
function(add_comparison_example EXAMPLE_NAME EXT)
    # Define our named keyword arguments
    set(options "")
    set(oneValueArgs RANKS)
    set(multiValueArgs LINK_LIBS RUN_ARGS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Fallback to default ranks if the user didn't specify the RANKS keyword
    if(NOT ARG_RANKS)
        set(ARG_RANKS ${BENCHMARK_RANKS})
    endif()

    # 1. Baseline
    add_executable(${EXAMPLE_NAME}_mpi ${EXAMPLE_NAME}_mpi.${EXT})
    target_include_directories(${EXAMPLE_NAME}_mpi PRIVATE ${MPI_C_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(${EXAMPLE_NAME}_mpi PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${ARG_LINK_LIBS})

    # 2. Harness Selection
    if("${EXT}" STREQUAL "c")
        set(CURRENT_HARNESS ${MDMP_HARNESS_C})
    else()
        set(CURRENT_HARNESS ${MDMP_HARNESS_CXX})
    endif()

    # 3. Optimized MDMP Build
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp
        COMMAND ${CURRENT_HARNESS} ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}_mdmp.${EXT}
                -I${CMAKE_SOURCE_DIR}/src
                -I/usr/include/mpich-x86_64
                /usr/lib64/mpich/lib/libmpi.so
                ${ARG_LINK_LIBS} # Linker flags go here
                -o ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}_mdmp.${EXT} mdmp_pragma_pass mdmp_runtime
        COMMENT "Compiling ${EXAMPLE_NAME}_mdmp using ${CURRENT_HARNESS}..."
    )

    add_custom_target(mdmp_example_${EXAMPLE_NAME} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp)

    # 4. === Execution Targets ===
    add_custom_target(run_example_${EXAMPLE_NAME}
        # Run Baseline (Inject ARG_RUN_ARGS at the end!)
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "========================================"
        COMMAND ${CMAKE_COMMAND} -E echo " BENCHMARK: ${EXAMPLE_NAME} -- Baseline MPI"
        COMMAND ${CMAKE_COMMAND} -E echo "========================================"
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ARG_RANKS} ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mpi ${ARG_RUN_ARGS}
        
        # Run Optimized (Inject ARG_RUN_ARGS at the end!)
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "========================================"
        COMMAND ${CMAKE_COMMAND} -E echo " BENCHMARK: ${EXAMPLE_NAME} -- MDMP Optimized"
        COMMAND ${CMAKE_COMMAND} -E echo "========================================"
        COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src:/usr/lib64/mpich/lib"
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ARG_RANKS} ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp ${ARG_RUN_ARGS}
        
        DEPENDS ${EXAMPLE_NAME}_mpi mdmp_example_${EXAMPLE_NAME}
        USES_TERMINAL 
    )

    add_dependencies(run_examples run_example_${EXAMPLE_NAME})
endfunction()

# ==============================================================================
# Register Example Programs Below
# ==============================================================================

# Heat equation (Requires the math library)
add_comparison_example(heat c LINK_LIBS -lm)

# Circuit satisfiability problem
add_comparison_example(satisfy cpp)

# Parallel Solution of the 1D Wave Equation
add_comparison_example(wave cpp)

# Example MPI benchmark code from the Grid codebase (Custom Ranks and Runtime Args!)
add_comparison_example(grid_halo cpp RANKS 16 RUN_ARGS --mpi 2.2.2.2)
