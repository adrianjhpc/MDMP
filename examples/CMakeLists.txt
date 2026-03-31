find_package(MPI REQUIRED)

set(MDMP_HARNESS_C "${CMAKE_SOURCE_DIR}/harness/mdmpc")
set(MDMP_HARNESS_CXX "${CMAKE_SOURCE_DIR}/harness/mdmpc++")

set(BENCHMARK_RANKS 256)

add_custom_target(run_examples
    COMMENT "Executing all benchmarks: MPI vs MDMP Imperative vs MDMP RegCom..."
)

# ------------------------------------------------------------------------------
# Helper Function: add_comparison_example
# Usage: add_comparison_example(<name> <ext> [RANKS <n>] [LINK_LIBS <libs...>] [RUN_ARGS <args...>])
# ------------------------------------------------------------------------------
function(add_comparison_example EXAMPLE_NAME EXT)
    set(options "")
    set(oneValueArgs RANKS)
    set(multiValueArgs LINK_LIBS RUN_ARGS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_RANKS)
        set(ARG_RANKS ${BENCHMARK_RANKS})
    endif()

    add_executable(${EXAMPLE_NAME}_mpi ${EXAMPLE_NAME}_mpi.${EXT})
    target_include_directories(${EXAMPLE_NAME}_mpi PRIVATE ${MPI_C_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(${EXAMPLE_NAME}_mpi PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${ARG_LINK_LIBS})

    if("${EXT}" STREQUAL "c")
        set(CURRENT_HARNESS ${MDMP_HARNESS_C})
        set(MPI_INC ${MPI_CXX_HEADER_DIR})
        set(MPI_LIB ${MPI_CXX_LIBRARIES})
    else()
        set(CURRENT_HARNESS ${MDMP_HARNESS_CXX})
        set(MPI_INC ${MPI_C_HEADER_DIR})
        set(MPI_LIB ${MPI_C_LIBRARIES})
    endif()

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp
        COMMAND ${CURRENT_HARNESS} ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}_mdmp.${EXT}
                -I${CMAKE_SOURCE_DIR}/src -I${MPI_INC}
                ${MPI_LIB} ${ARG_LINK_LIBS}
                -o ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}_mdmp.${EXT} mdmp_compiler_pass mdmp_runtime
        COMMENT "Compiling ${EXAMPLE_NAME}_mdmp (Imperative)..."
    )
    add_custom_target(target_${EXAMPLE_NAME}_mdmp ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp)

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp_regcom
        COMMAND ${CURRENT_HARNESS} ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}_mdmp_regcom.${EXT}
                -I${CMAKE_SOURCE_DIR}/src -I${MPI_INC}
                ${MPI_LIB} ${ARG_LINK_LIBS}
                -o ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp_regcom
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}_mdmp_regcom.${EXT} mdmp_compiler_pass mdmp_runtime
        COMMENT "Compiling ${EXAMPLE_NAME}_mdmp_regcom (Declarative)..."
    )
    add_custom_target(target_${EXAMPLE_NAME}_mdmp_regcom ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp_regcom)

    set(RUN_MPI ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ARG_RANKS} ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mpi ${ARG_RUN_ARGS})
    set(RUN_MDMP ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ARG_RANKS} ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp ${ARG_RUN_ARGS})
    set(RUN_REGCOM ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ARG_RANKS} ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_NAME}_mdmp_regcom ${ARG_RUN_ARGS})
    set(MDMP_ENV "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src:/usr/lib64/mpich/lib:$ENV{LD_LIBRARY_PATH}")

    add_custom_target(run_example_${EXAMPLE_NAME}
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "------------------------------------------------"
        COMMAND ${CMAKE_COMMAND} -E echo " BENCHMARK: ${EXAMPLE_NAME} -- Baseline MPI"
        COMMAND ${CMAKE_COMMAND} -E echo "------------------------------------------------"
        COMMAND ${RUN_MPI}

        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "------------------------------------------------"
        COMMAND ${CMAKE_COMMAND} -E echo " BENCHMARK: ${EXAMPLE_NAME} -- MDMP Imperative"
        COMMAND ${CMAKE_COMMAND} -E echo "------------------------------------------------"
        COMMAND ${CMAKE_COMMAND} -E env ${MDMP_ENV} ${RUN_MDMP}

        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "------------------------------------------------"
        COMMAND ${CMAKE_COMMAND} -E echo " BENCHMARK: ${EXAMPLE_NAME} -- MDMP Declarative (RegCom)"
        COMMAND ${CMAKE_COMMAND} -E echo "------------------------------------------------"
        COMMAND ${CMAKE_COMMAND} -E env ${MDMP_ENV} ${RUN_REGCOM}

        DEPENDS ${EXAMPLE_NAME}_mpi target_${EXAMPLE_NAME}_mdmp target_${EXAMPLE_NAME}_mdmp_regcom
        USES_TERMINAL
        VERBATIM
    )

    add_dependencies(run_examples run_example_${EXAMPLE_NAME})
endfunction()

# ==============================================================================
# Register example programs below
# ==============================================================================

add_comparison_example(heat c LINK_LIBS -lm)
add_comparison_example(heat_2d cpp)
add_comparison_example(satisfy cpp)
add_comparison_example(wave cpp)
add_comparison_example(grid_halo cpp RANKS 16 RUN_ARGS --mpi 2.2.2.2)
add_comparison_example(nbody cpp)
add_comparison_example(graph cpp)
add_comparison_example(wavefront cpp)
add_comparison_example(stencil cpp)
