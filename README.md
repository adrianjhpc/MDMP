# MDMP: Managed Data Message Passing
MDMP is an alternative distributed memory programming approach that recognises some of the challenges of using an entirely prescriptive approach, such as MPI, for distributed applications. It is a hybrid LLVM compiler pass and C++ runtime library designed to automatically optimise distributed-memory HPC applications. By analyzing memory dependencies at compile-time, MDMP safely decouples network communication from local computation, transforming rigid imperative MPI calls into highly asynchronous, overlapping workloads.


Specifically, MDMP is designed to allow more variation in the granularity and timing of communications between collaborating processes, both to optimise the use of the network for a given program, and to reduce the overhead of implementing th message passing approach for programmers.

It is built on a compiler directives model, similar to the approach used by OpenMP, meaning a compiler generates the message passing functionality based on directives added to the program by the developer. In theory this should enable a seial version of the distributed memory program to be created and run without the directives being processed, as well as a distributed memory version with the directives.

MDMP is designed to work alongside existing MPI functionality, so it can be added incrementally to existing MPI programs as well as being used to develop distributed memory approaches from scratch.

MDMP is not designed to replace MPI entirely, indeed its functionality is built on the MPI library. It is more focussed on providing an alternative programming approach to using MPI that can enable more optimised, or at least more varied, communication patterns to be implemented without requiring significant code changes by developers and users.


## Key Features

* **LLVM Compiler Pass (JIT Optimisation):** Utilizes strict Alias Analysis, Dominance Trees, and Loop Information to automatically hoist network initiation instructions and push synchronisation (`wait`) calls as far down the control flow as physically safe.
* **Declarative Inspector-Executor API:** Replaces the classic "Compute-Then-Communicate" bottleneck. Users can register thousands of network intent calls (`mdmp_register_send`, `mdmp_register_gather`) and execute them concurrently via a single `mdmp_commit()`.
* **Automated Batching:** The runtime automatically sorts and coalesces concurrent messages to identical peers using `MPI_Type_create_hindexed`, drastically reducing network hardware contention.
* **Background Progress Engine:** A thread-safe, mutex-guarded `std::thread` continuously polls `MPI_Iprobe` to pump the network while the main CPU cores are locked in heavy compute loops.
* **Legacy Compatibility:** Drop-in imperative wrappers (`mdmp_send`, `mdmp_recv`) allow for immediate performance gains on legacy codebases (like Gadget-2) without algorithmic rewrites.

---

## Architecture

MDMP operates in two phases:

1. **Compile-Time (Static Analysis):** The LLVM pass (`mdmp_compiler_pass.cpp`) searches for `__mdmp_marker_` injected by the C headers. It tracks the exact memory locations of send/recv buffers and walks the Basic Blocks to hoist initiation and inject `mdmp_wait` right before the CPU physically requires the data. It natively prevents "inner-loop poisoning" by anchoring wait states to loop latches.
   
2. **Run-Time (Dynamic Execution):**
   The C++ backend (`mdmp_runtime.cpp`) maps the LLVM-injected markers to actual MPI instructions. It features unbounded request ID tracking via `std::unordered_map` and gracefully handles dynamic 0-byte transfer fallbacks.

---

## Building MDMP

### Prerequisites
* LLVM / Clang (Version 14.0+ recommended)
* An MPI implementation featuring
* CMake 3.10+ 

### Compilation

```bash
git clone [https://github.com/adrianjhpc/MDMP.git](https://github.com/adrianjhpc/MDMP.git)
cd MDMP
mkdir build && cd build
cmake ..
make -j
```

This will produce the runtime library (`libmdmp_runtime.so`) and the LLVM plugin (MDMPPass.so).

### Usage & Integration
1. The Compiler Wrapper
To compile your application with MDMP, use the provided compiler wrapper or pass the plugin directly to Clang:

```bash
clang -fpass-plugin=/path/to/MDMPPass.so -O3 -c my_app.c
```

2. Available Compiler Flags
MDMP includes custom LLVM flags to toggle experimental optimisations. Use the `-mllvm` prefix to pass these down:

`-mdmp-progress`: Enables JIT progress injection. The compiler will inject a 64-iteration modulo check inside leaf loops to manually call `mdmp_maybe_progress(`). We make this selectable because it can have a performance impact as well as a performance benefit.

Example:

```bash
clang -fpass-plugin=/path/to/MDMPPass.so -mllvm -mdmp-progress -O3 -c my_app.c
```

3. API Examples
Imperative (Legacy Integration)
Provides immediate asynchronous overlap by allowing the compiler to slide the implicit wait state down the block.

```c
#include "mdmp_interface.h"

// The LLVM pass will automatically hoist this send and delay the wait
mdmp_send(send_buf, 100, MDMP_FLOAT, sizeof(float)*100, my_rank, dest_rank, 0);

heavy_local_computation(); // <--- Network transfers during this work

// LLVM automatically injects mdmp_wait() here, right before buffer reuse
```

Declarative (Modern Stencils / Mesh Refinement)
Perfect for halo exchanges. Eliminates ID tracking and loop-leakage by pushing the entire batch to the runtime.

```c
#include "mdmp_interface.h"

MDMP_COMMREGION_BEGIN();
for (int dir = 0; dir < 6; dir++) {
    mdmp_register_send(halos[dir], ...);
    mdmp_register_recv(halos[dir], ...);
}
mdmp_commit(); // Fires all requests optimally, coalescing identical targets

heavy_inner_cell_math();

MDMP_COMMREGION_END(); // Safely waits for all pending declarative batches
```
## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
