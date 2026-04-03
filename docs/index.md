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

If MPI has already been initialized by the application, MDMP will attach to the existing MPI environment. Otherwise, MDMP initializes MPI itself.

A minimal imperative example

```cpp
#include <vector>
#include "mdmp_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();

    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    if (size < 2) {
        MDMP_COMM_FINAL();
        return 0;
    }

    std::vector<double> sendbuf(1024, rank);
    std::vector<double> recvbuf(1024, 0.0);

    int dest = (rank + 1) % size;
    int src  = (rank - 1 + size) % size;

    MDMP_RECV(recvbuf.data(), 1024, rank, src, 0);
    MDMP_SEND(sendbuf.data(), 1024, rank, dest, 0);

    // Useful computation can happen here while communication is in flight

    MDMP_COMM_FINAL();
    return 0;
}
```

The LLVM pass will transform these calls into non-blocking runtime operations and place waits where required.

A minimal declarative example

```cpp
#include <vector>
#include "mdmp_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();

    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    if (size < 2) {
        MDMP_COMM_FINAL();
        return 0;
    }

    std::vector<double> sendbuf(1024, rank);
    std::vector<double> recvbuf(1024, 0.0);

    int dest = (rank + 1) % size;
    int src  = (rank - 1 + size) % size;

    MDMP_COMM_REGION_BEGIN();

    MDMP_REGISTER_RECV(recvbuf.data(), 1024, rank, src, 0);
    MDMP_REGISTER_SEND(sendbuf.data(), 1024, rank, dest, 0);

    MDMP_COMMIT();

    MDMP_COMM_REGION_END();

    MDMP_COMM_FINAL();
    return 0;
}
```

In this mode, MDMP treats the region as a batch and may coalesce multiple communication operations into fewer MPI requests.

## When to use each model

Use Imperative MDMP when:

 - the communication pattern is already explicit
 - you want minimal code changes from MPI-style logic
 - messages are tied to specific program points
 - you want the compiler to automatically overlap communication with nearby compute

Use Declarative MDMP when:

 - a region contains many small communication operations
 - messages can be registered first and launched together
 - batching or coalescing may reduce overhead
 - you want communication regions to behave like a higher-level schedule

## Performance philosophy

MDMP does not make all MPI codes faster automatically. It performs best when:

 - communication can be started significantly before data is needed
 - there is useful work available while messages are in flight
 - communication patterns repeat across iterations
 - batching and coalescing reduce request overhead

In some applications, especially highly compute-heavy kernels with very small messages, the performance difference versus hand-written MPI may be small. In others, particularly latency-sensitive or irregular codes, MDMP can significantly improve overlap and reduce communication overhead.

## Next steps

 - See [Getting Started](getting-started.md) for build and compile instructions
 - See Imperative MDMP for point-to-point and collective usage
 - See Declarative MDMP for communication regions and batching
 - See Performance Notes for tuning advice and expected behavior

