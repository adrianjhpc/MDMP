# Welcome to MDMP

**MDMP** (Multi-Paradigm Message Passing) is a compiler-assisted runtime built on top of MPI. It is designed to completely automate the most difficult aspects of High-Performance Computing: network latency mitigation and computation/communication overlap.

## The Core Philosophy
No single communication model fits all physics simulations. MDMP provides two paradigms:

1. **The Imperative Model:** Best for unstructured grids and graph analytics. You write standard blocking calls, and an LLVM pass automatically injects non-blocking wait routines to maximize CPU/Network overlap.
2. **The Declarative Model (RegCom):** Best for N-Body simulations and scattered data. You wrap your communication in a `COMMREGION`. The runtime dynamically generates `MPI_Type_create_hindexed` hardware datatypes, firing a single zero-copy payload.

## Initialization & Environment
All MDMP programs must initialize and finalize the runtime environment.

```cpp
#include "mdmp_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();
    
    // ... application code ...
    
    MDMP_COMM_FINAL();
    return 0;
}
