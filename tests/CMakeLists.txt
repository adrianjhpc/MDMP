cmake_minimum_required(VERSION 3.13)

set(MDMPCXX "${CMAKE_CURRENT_SOURCE_DIR}/../harness/mdmpc++")

string(REPLACE ";" " -I" MPI_INC_FLAGS "${MPI_CXX_INCLUDE_DIRS}")
if(MPI_INC_FLAGS)
    set(MPI_INC_FLAGS "-I${MPI_INC_FLAGS}")
else()
    set(MPI_INC_FLAGS "-I${MPI_CXX_HEADER_DIR}")
endif()

# Find MPI to locate the mpirun/mpiexec executable
find_package(MPI REQUIRED)

# We define how many ranks CTest should use to run your test suite
set(TEST_NUM_RANKS 7)

function(add_mdmp_test TEST_NAME SOURCE_FILE)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
        COMMAND ${MDMPCXX} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}
                ${MPI_INC_FLAGS} ${MPI_LIBRARIES}
                -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
        # Rebuild if the source file, the pass, or the runtime changes!
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE} mdmp_compiler_pass mdmp_runtime
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
add_mdmp_test(mdmp_test_rank_selection_regcom test_rank_selection_mdmp_regcom.cpp)
add_mdmp_test(mdmp_test_out_of_bounds_rank_selection test_out_of_bounds_rank_selection_mdmp.cpp)
add_mdmp_test(mdmp_test test_mdmp.cpp)
add_mdmp_test(mdmp_test_regcom test_mdmp_regcom.cpp)
add_mdmp_test(mdmp_test_periodic_boundaries test_periodic_mdmp.cpp)
add_mdmp_test(mdmp_test_periodic_boundaries_regcom test_periodic_mdmp_regcom.cpp)
add_mdmp_test(mdmp_test_reduce test_reduce_mdmp.cpp)
add_mdmp_test(mdmp_test_reduce_max test_reduce_max_mdmp.cpp)
add_mdmp_test(mdmp_test_reduce_min test_reduce_min_mdmp.cpp)
add_mdmp_test(mdmp_test_reduce_non_zero_root test_reduce_non_zero_root_mdmp.cpp)
add_mdmp_test(mdmp_test_reduce_regcom test_reduce_mdmp_regcom.cpp)
add_mdmp_test(mdmp_test_reduce_non_zero_root_regcom test_reduce_non_zero_root_mdmp_regcom.cpp)
add_mdmp_test(mdmp_test_gather test_gather_mdmp.cpp)
add_mdmp_test(mdmp_test_gather_regcom test_gather_mdmp_regcom.cpp)
add_mdmp_test(mdmp_test_allreduce test_mdmp_allreduce.cpp)
add_mdmp_test(mdmp_test_allreduce_mpi_sum test_mdmp_allreduce_mpi_sum.cpp )
add_mdmp_test(mdmp_test_allreduce_min test_mdmp_allreduce_min.cpp)
add_mdmp_test(mdmp_test_allreduce_max test_mdmp_allreduce_max.cpp)
add_mdmp_test(mdmp_test_allreduce_regcom test_mdmp_allreduce_regcom.cpp)
add_mdmp_test(mdmp_test_allreduce_min_regcom test_mdmp_allreduce_min_regcom.cpp)
add_mdmp_test(mdmp_test_allreduce_max_regcom test_mdmp_allreduce_max_regcom.cpp)
add_mdmp_test(mdmp_test_allgather test_mdmp_allgather.cpp)
add_mdmp_test(mdmp_test_allgather_regcom test_mdmp_allgather_regcom.cpp)
add_mdmp_test(mdmp_test_bcast test_mdmp_bcast.cpp)
add_mdmp_test(mdmp_test_bcast_regcom test_mdmp_bcast_regcom.cpp)
add_mdmp_test(mdmp_test_regcom_pack test_mdmp_regcom_pack.cpp)
add_mdmp_test(mdmp_test_bcast_race test_mdmp_bcast_race.cpp)
add_mdmp_test(mdmp_Test_bcast_race_global_struct test_mdmp_bcast_race_global_struct.cpp)
add_mdmp_test(mdmp_test_multitype test_mdmp_multitype.cpp)
add_mdmp_test(mdmp_test_abort test_mdmp_abort.cpp)
# Tell CTest to expect a non-zero exit code (because MPI_Abort forcefully kills the process)
set_tests_properties(mdmp_test_abort PROPERTIES WILL_FAIL TRUE)
