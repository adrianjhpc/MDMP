# Getting Started

This page explains how to build, compile, and run MDMP applications.

## Requirements

MDMP requires:

- an MPI implementation
- LLVM/Clang with support for the MDMP pass
- CMake
- a C++ compiler compatible with the MDMP runtime and pass

## MDMP programming model

Applications include:

```cpp
#include "mdmp_interface.h"
```

and must call:

 - `MDMP_COMM_INIT()`
 - ` MDMP_COMM_FINAL()`

before and after MDMP communication.

## Basic workflow

An MDMP application is compiled in two parts:

 - the program source
 - the MDMP runtime and compiler pass

The MDMP compiler pass rewrites MDMP markers into runtime calls and inserts communication scheduling logic.
Typical application skeleton

```cpp
#include "mdmp_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();

    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    // Computation and MDMP communication

    MDMP_COMM_FINAL();
    return 0;
}
```

## Running

MDMP programs are launched with the normal MPI launcher for your system, for example:

`mpirun -n 4 ./my_mdmp_program`

or:

`srun -n 4 ./my_mdmp_program`

depending on your cluster environment.

