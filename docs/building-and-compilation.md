# Build and Compilation

MDMP consists of:

- a runtime library
- an LLVM compiler pass
- an interface header for user code

## Compilation model

MDMP source-level calls are lowered by the compiler pass into runtime calls such as:

- `mdmp_send`
- `mdmp_recv`
- `mdmp_commit`
- `mdmp_wait`

The pass also inserts additional overlap-related logic, including communication hoisting and wait placement.

## Typical build process

A typical MDMP build has two steps:

1. build the MDMP runtime and compiler pass
2. compile the application with the MDMP pass enabled

Depending on your local toolchain and CMake setup, the exact commands may differ.

## Notes

MDMP is implemented as a compiler-assisted model, so compiling without the pass may leave marker functions in place rather than the intended optimized runtime behavior.

## Recommended practice

Use the provided CMake build system and examples as the reference configuration for compiling new applications.


