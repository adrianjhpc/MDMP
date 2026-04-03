# Welcome to MDMP

**MDMP** (**M**ulti-**P**aradigm **M**essage **P**assing) is a compiler-assisted communication layer built on top of MPI. It is designed to help HPC applications overlap communication and computation automatically, while preserving a simple programming model.

MDMP provides two communication styles:

1. **Imperative MDMP**  
   Write communication explicitly with `MDMP_SEND`, `MDMP_RECV`, and the MDMP collective calls.  
   The compiler pass lowers these to non-blocking runtime calls, hoists communication initiation when safe, and inserts waits only where the communicated data is actually needed.

2. **Declarative MDMP (Communication Regions)**  
   Register communication operations with `MDMP_REGISTER_*` inside a communication region, then call `MDMP_COMMIT()`.  
   This allows the runtime to batch and coalesce communication, reorder safely within the region, and reduce communication overhead.

MDMP is intended for applications where message latency and communication scheduling are difficult to manage by hand, including:

- structured and unstructured stencil codes
- sparse or scattered communication patterns
- irregular graph and mesh workloads
- iterative solvers and simulation codes

---

## Core idea

Traditional MPI programming often forces the programmer to manually decide:

- when communication should start
- how early it can be issued safely
- where completion must be enforced
- how to overlap communication with useful work

MDMP automates much of this process.

At compile time, the MDMP LLVM pass:

- replaces MDMP markers with runtime calls
- tracks communicated memory regions
- hoists communication initiation upward when safe
- inserts waits close to the first true consumer or clobber
- optionally injects progress calls in long-running loops

At runtime, MDMP:

- launches non-blocking MPI operations
- batches declarative communication regions
- coalesces multiple registered operations when possible
- routes waits efficiently for imperative and declarative requests

---

## Initialization

All MDMP programs must initialize and finalize the runtime.

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
```
