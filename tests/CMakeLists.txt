cmake_minimum_required(VERSION 3.13)

set(MDMPCXX "${CMAKE_CURRENT_SOURCE_DIR}/../harness/mdmpc++")

# ==========================================
# Fortran Testing Setup
# ==========================================
# Point to a new Fortran harness script
set(MDMPF90 "${CMAKE_CURRENT_SOURCE_DIR}/../harness/mdmpfortran")

function(add_mdmp_fortran_test TEST_NAME)
    # Only create the test if Fortran support is enabled at the top level
    if(NOT MDMP_ENABLE_FORTRAN)
        return()
    endif()
    
    set(ABSOLUTE_SOURCES "")
    foreach(src_file IN LISTS ARGN)
        list(APPEND ABSOLUTE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
    endforeach()

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
        COMMAND ${MDMPF90} ${ABSOLUTE_SOURCES}
                ${MPI_INC_FLAGS} 
                -I${CMAKE_BINARY_DIR}/include/mdmp_fortran # Location of mdmp_interface.mod
                -L${CMAKE_BINARY_DIR}/src -lmdmp_runtime_f # Link the Fortran runtime
                ${MPI_LIBRARIES}
                -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
        # Depend on both the C and Fortran runtimes
        DEPENDS ${ABSOLUTE_SOURCES} mdmp_compiler_pass mdmp_runtime mdmp_runtime_f
        COMMENT "Compiling ${TEST_NAME} using mdmpfortran harness..."
    )

    add_custom_target(${TEST_NAME}_target ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
    )

    add_test(NAME ${TEST_NAME}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${TEST_NUM_RANKS} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
    )

    set_tests_properties(${TEST_NAME} PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src:/usr/lib64/mpich/lib:$ENV{LD_LIBRARY_PATH}"
    )
endfunction()

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

# CHANGED: Removed SOURCE_FILE from the signature. 
# Everything after TEST_NAME goes into ${ARGN}.
function(add_mdmp_test TEST_NAME)
    
    # 1. Collect all source files and make their paths absolute
    set(ABSOLUTE_SOURCES "")
    foreach(src_file IN LISTS ARGN)
        list(APPEND ABSOLUTE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
    endforeach()

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
        # 2. Pass the entire list of source files to mdmpc++
        COMMAND ${MDMPCXX} ${ABSOLUTE_SOURCES}
                ${MPI_INC_FLAGS} ${MPI_LIBRARIES}
                -o ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
        # 3. Depend on ALL source files, so any edit triggers a rebuild
        DEPENDS ${ABSOLUTE_SOURCES} mdmp_compiler_pass mdmp_runtime
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
add_mdmp_test(mdmp_test_interprocedural test_mdmp_interprocedural.cpp)
add_mdmp_test(mdmp_test_deadlock test_mdmp_deadlock.cpp)
add_mdmp_test(mdmp_test_deadlock_noinline test_mdmp_deadlock_noinline.cpp)
add_mdmp_test(mdmp_test_deadlock_lto test_mdmp_deadlock_lto_procedures.cpp test_mdmp_deadlock_lto.cpp)
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

if(MDMP_ENABLE_FORTRAN)
    add_mdmp_fortran_test(mdmp_test_fortran test_mdmp.f90)
endif()
