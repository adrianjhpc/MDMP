# CMakeLists.txt for building with LLVM support
cmake_minimum_required(VERSION 3.13)

# Include LLVM directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})

# Add definitions from LLVM
add_definitions(${LLVM_DEFINITIONS})


# Ensure Analysis and ScalarOpts are included
llvm_map_components_to_libnames(llvm_libs 
  Support 
  Core 
  Analysis 
  ScalarOpts 
  TransformUtils
)

add_library(mdmp_runtime SHARED
    mdmp_runtime.cpp
)

# Create the MDMP compiler pass library
add_library(mdmp_compiler_pass MODULE
    mdmp_compiler_pass.cpp
)
target_link_libraries(mdmp_compiler_pass PRIVATE
)

# Set library properties
set_target_properties(mdmp_compiler_pass PROPERTIES
    PREFIX ""
    OUTPUT_NAME "MDMP"
)

# Install headers
install(FILES 
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

# Include MPI headers for the runtime
target_include_directories(mdmp_runtime PRIVATE ${MPI_INCLUDE_PATH})

# Link the runtime against MPI
target_link_libraries(mdmp_runtime PRIVATE ${MPI_LIBRARIES})

# Install the runtime library alongside the pass
install(TARGETS mdmp_runtime
    LIBRARY DESTINATION lib
    COMPONENT mdmp_runtime
)

if(MDMP_ENABLE_FORTRAN)
    # Tell CMake where to output the compiled .mod file
    set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include/mdmp_fortran)

    # Create the dedicated Fortran library
    add_library(mdmp_runtime_f SHARED
        mdmp_interface.f90
    )

    # Link the Fortran interface to your base C/C++ runtime
    target_link_libraries(mdmp_runtime_f PRIVATE mdmp_runtime)

    # Install the Fortran shared library
    install(TARGETS mdmp_runtime_f
        LIBRARY DESTINATION lib
        COMPONENT mdmp_runtime
    )

    # Install the generated .mod file so user code can "use mdmp_interface"
    install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/
        DESTINATION include
        COMPONENT Headers
    )
    
    # Optionally install the raw .f90 file just in case users want to 
    # compile it directly into their own source tree instead of linking
    install(FILES mdmp_interface.f90
        DESTINATION include
        COMPONENT Headers
    )
endif()
