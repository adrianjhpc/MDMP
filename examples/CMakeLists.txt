# ==============================================================================
# MDMP Examples CMake Configuration
# ==============================================================================

# Find standard MPI for the baseline comparison program
find_package(MPI REQUIRED)

# ------------------------------------------------------------------------------
# 1. Baseline: Standard MPI Heat Equation
# ------------------------------------------------------------------------------
add_executable(heat_mpi heat_mpi.c)
target_include_directories(heat_mpi PRIVATE ${MPI_C_INCLUDE_DIRS})
target_link_libraries(heat_mpi PRIVATE ${MPI_C_LIBRARIES} m) # 'm' is for the math library (sin)

# ------------------------------------------------------------------------------
# 2. Optimized: MDMP Heat Equation
# ------------------------------------------------------------------------------
# We compile this using your custom harness to ensure the LLVM pass runs,
# the CFG hoister executes, and the runtime is linked natively.

set(MDMP_HARNESS "${CMAKE_SOURCE_DIR}/harness/mdmpc++")

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/heat_mdmp
    COMMAND ${MDMP_HARNESS} ${CMAKE_CURRENT_SOURCE_DIR}/heat_mdmp.c 
            -I${CMAKE_SOURCE_DIR}/src 
            -I/usr/include/mpich-x86_64 
            /usr/lib64/mpich/lib/libmpi.so 
            -lm  # Link the math library
            -o ${CMAKE_CURRENT_BINARY_DIR}/heat_mdmp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/heat_mdmp.c mdmp_pragma_pass mdmp_runtime
    COMMENT "Compiling heat_mdmp using mdmpc++ harness..."
)

# Create a target so `make` knows to trigger the custom command above
add_custom_target(mdmp_example_heat ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/heat_mdmp)
