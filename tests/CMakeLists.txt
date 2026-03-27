cmake_minimum_required(VERSION 3.13)

set(MDMPCXX "${CMAKE_CURRENT_SOURCE_DIR}/../harness/mdmpc++")

string(REPLACE ";" " -I" MPI_INC_FLAGS "${MPI_CXX_INCLUDE_DIRS}")
if(MPI_INC_FLAGS)
    set(MPI_INC_FLAGS "-I${MPI_INC_FLAGS}")
else()
    set(MPI_INC_FLAGS "-I/usr/include/mpich-x86_64") # Safe fallback
endif()

# Find MPI to locate the mpirun/mpiexec executable
find_package(MPI REQUIRED)

# We define how many ranks CTest should use to run your test suite
set(TEST_NUM_RANKS 4)

function(add_mdmp_test TEST_NAME SOURCE_FILE)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
        # REMOVED ${MPI_CXX_COMPILE_FLAGS} below:
        COMMAND ${MDMPCXX} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}
                ${MPI_INC_FLAGS} ${MPI_LIBRARIES}
                -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
        # Rebuild if the source file, the pass, or the runtime changes!
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} mdmp_pragma_pass mdmp_runtime
        COMMENT "Compiling ${TEST_NAME} using mdmpc++ harness..."
    )

    # Attach the custom command to a target so 'make' knows to build it
    add_custom_target(${TEST_NAME}_target ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
    )

    # Register the executable with CTest, wrapping it in mpiexec/mpirun
    add_test(NAME ${TEST_NAME} 
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NUM_RANKS} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
    )

    # Inject the LD_LIBRARY_PATH so CTest can find your custom runtime during execution
    set_tests_properties(${TEST_NAME} PROPERTIES 
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src:/usr/lib64/mpich/lib:$ENV{LD_LIBRARY_PATH}"
    )
endfunction()

# Enable CTest runner
enable_testing()

add_mdmp_test(mdmp_test_rank_selection test_rank_selection_mdmp.cpp)
add_mdmp_test(mdmp_test_out_of_bounds_rank_selection test_out_of_bounds_rank_selection_mdmp.cpp)
add_mdmp_test(mdmp_test test_mdmp.cpp)
add_mdmp_test(mdmp_test_periodic_boundaries test_periodic_mdmp.cpp)
add_mdmp_test(mdmp_test_reduce test_reduce_mdmp.cpp)
add_mdmp_test(mdmp_test_reduce_non_zero_root test_reduce_non_zero_root_mdmp.cpp)
add_mdmp_test(mdmp_test_gather test_gather_mdmp.cpp)
