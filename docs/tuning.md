# MDMP Progress Tuning Profiles

MDMP’s compiler pass can inject communication progress calls to help nonblocking MPI requests complete while useful work is still running. This is most beneficial for large, irregular applications, but can add overhead for small, regular kernels.

MDMP therefore supports different **compile-time progress tuning profiles**.

> These settings are read by the **compiler pass when the application is compiled**, not at runtime.  
> You must set them **before building the application**.

---

# 1. Compile-time progress tuning variables

The following environment variables control compile-time progress insertion.

## `MDMP_PROGRESS_DEBUG`

Enable compiler-side diagnostics showing where progress calls are inserted.

### Example

```bash
export MDMP_PROGRESS_DEBUG=1
```

### Typical values

 - 0 → disabled
 - 1 → enabled

##`MDMP_PROGRESS_RELAXED`

Enable relaxed progress insertion heuristics beyond the most conservative exact loop-window matching.
Example

```bash
export MDMP_PROGRESS_RELAXED=1
```
Typical values

 - 0 → conservative mode
 - 1 → relaxed mode

## `MDMP_PROGRESS_PERIOD`

Controls the polling period for loop-based progress insertion.

A lower value means progress is checked more often.
A higher value reduces overhead but may reduce overlap.

### Example
```bash
export MDMP_PROGRESS_PERIOD=64
```
### Typical values

 - 64 → aggressive / irregular-application tuning
 - 4096 or 16384 → conservative / regular-kernel tuning

## `MDMP_PROGRESS_MAX_CALLSITES`

Maximum number of fallback progress callsites inserted per function.

These are used when loop-based progress insertion fails to find suitable locations.
### Example
```bash
export MDMP_PROGRESS_MAX_CALLSITES=16
```
### Typical values

  - 0 → disable callsite fallback completely
  - 8 or 16 → enable aggressive fallback in irregular codes

## `MDMP_PROGRESS_MAX_DEEP_LOOPS`

Maximum number of “forced deep leaf loop” fallback progress sites inserted per function.

These are a last-resort fallback for complex functions where no other progress site can be found.
### Example

```bash
export MDMP_PROGRESS_MAX_DEEP_LOOPS=2
```
### Typical values

 - 0 → disable forced deep-loop fallback
 - 1 or 2 → enable for irregular applications

## `MDMP_PROGRESS_AGGR_MIN_REQS`

Minimum number of in-flight requests in a function before aggressive progress fallback is considered profitable.
### Example

```bash
export MDMP_PROGRESS_AGGR_MIN_REQS=8
```

## `MDMP_PROGRESS_AGGR_MIN_BYTES`

Minimum total precise communicated byte volume in a function before aggressive fallback is considered profitable.
### Example
```bash
export MDMP_PROGRESS_AGGR_MIN_BYTES=32768
```

## `MDMP_PROGRESS_AGGR_UNKNOWN_MIN_REQS`

If some communicated sizes are not statically known, this sets the minimum number of requests required before aggressive fallback is allowed.
### Example

```bash
export MDMP_PROGRESS_AGGR_UNKNOWN_MIN_REQS=12
```

2. Recommended tuning profiles

## Conservative profile

Use this for:

 - simple stencils
 - structured halo exchanges
 - very small communication phases
 - microbenchmarks
 - kernels with many timesteps and tiny per-step communication

This profile minimizes progress overhead.

```bash
export MDMP_PROGRESS_DEBUG=0
export MDMP_PROGRESS_RELAXED=0
export MDMP_PROGRESS_PERIOD=16384
export MDMP_PROGRESS_MAX_CALLSITES=0
export MDMP_PROGRESS_MAX_DEEP_LOOPS=0
export MDMP_PROGRESS_AGGR_MIN_REQS=999999
export MDMP_PROGRESS_AGGR_MIN_BYTES=999999999
export MDMP_PROGRESS_AGGR_UNKNOWN_MIN_REQS=999999
```

### Interpretation

This effectively disables aggressive fallback progress insertion and keeps any remaining loop-based progress extremely infrequent.
Aggressive irregular-application profile

Use this for:

 - tree-based or graph-based computations
 - helper-heavy communication/computation phases
 - codes where overlap opportunities are hard to detect with strict loop matching alone

```bash
export MDMP_PROGRESS_DEBUG=0
export MDMP_PROGRESS_RELAXED=1
export MDMP_PROGRESS_PERIOD=64
export MDMP_PROGRESS_MAX_CALLSITES=16
export MDMP_PROGRESS_MAX_DEEP_LOOPS=2
export MDMP_PROGRESS_AGGR_MIN_REQS=8
export MDMP_PROGRESS_AGGR_MIN_BYTES=32768
export MDMP_PROGRESS_AGGR_UNKNOWN_MIN_REQS=12
```

### Interpretation

This enables more aggressive fallback progress insertion in functions that look large or communication-heavy enough to justify it.
Debug / diagnostics profile

Use this when investigating where progress sites are inserted.

```bash
export MDMP_PROGRESS_DEBUG=1
export MDMP_PROGRESS_RELAXED=1
export MDMP_PROGRESS_PERIOD=64
export MDMP_PROGRESS_MAX_CALLSITES=16
export MDMP_PROGRESS_MAX_DEEP_LOOPS=2
export MDMP_PROGRESS_AGGR_MIN_REQS=8
export MDMP_PROGRESS_AGGR_MIN_BYTES=32768
export MDMP_PROGRESS_AGGR_UNKNOWN_MIN_REQS=12
```

This will print compiler-side diagnostics such as:

 - number of request windows found
 - number of leaf and non-leaf loops
 - whether aggressive fallback is enabled
 - whether progress was inserted via:
      - exact leaf-loop matching
      - relaxed loop matching
      - callsite fallback
      -  forced deep-leaf fallback

3. How to use these profiles

Because these variables affect the compiler pass, set them before compiling the application.
Example: compiling a small stencil benchmark conservatively
```bash
export MDMP_PROGRESS_RELAXED=0
export MDMP_PROGRESS_PERIOD=16384
export MDMP_PROGRESS_MAX_CALLSITES=0
export MDMP_PROGRESS_MAX_DEEP_LOOPS=0
export MDMP_PROGRESS_AGGR_MIN_REQS=999999
export MDMP_PROGRESS_AGGR_MIN_BYTES=999999999
export MDMP_PROGRESS_AGGR_UNKNOWN_MIN_REQS=999999

make clean
make
```
### Example: compiling GADGET with aggressive fallback enabled
```bash
export MDMP_PROGRESS_RELAXED=1
export MDMP_PROGRESS_PERIOD=64
export MDMP_PROGRESS_MAX_CALLSITES=16
export MDMP_PROGRESS_MAX_DEEP_LOOPS=2
export MDMP_PROGRESS_AGGR_MIN_REQS=8
export MDMP_PROGRESS_AGGR_MIN_BYTES=32768
export MDMP_PROGRESS_AGGR_UNKNOWN_MIN_REQS=12

make clean
make
```
4. Runtime variables are different

Do not confuse the compile-time pass controls above with runtime-only MDMP variables.

## Runtime-only variables

These are read when the program executes:

 - `MDMP_PROFILE`
 - `MDMP_DEBUG`

### Example

```bash
export MDMP_PROFILE=1
srun ./my_mdmp_application
```

These do not change the generated code. They only affect runtime diagnostics or profiling.

5. Practical guidance

## If a simple benchmark slows down
Try the conservative profile first.

This is especially important for:
 - very regular codes
 - tiny halo exchanges
 - many millions of timesteps
 - cases where a few microseconds of per-step overhead matter

## If a large irregular application shows no overlap

Try the aggressive profile and enable `MDMP_PROFILE=1` at runtime to see whether:

 - progress calls are actually happening
 - requests are being tested while useful work is running
 - waits are completing earlier

## If aggressive mode hurts performance

That usually means either:

 - progress is being inserted too often, or
 - it is being inserted in the wrong places

In that case:

 - raise `MDMP_PROGRESS_PERIOD`
 - lower `MDMP_PROGRESS_MAX_CALLSITES`
 - lower `MDMP_PROGRESS_MAX_DEEP_LOOPS`
 - or increase the aggressive profitability thresholds

6. Summary
## Recommended default

For general use, a conservative default is safer.

## Recommended strategy

 - use conservative progress for simple regular kernels
 - use aggressive progress only for irregular codes that need it
 - use `MDMP_PROFILE=1` to validate whether progress insertion is actually helping

This two-profile approach avoids paying aggressive progress overhead in applications where it cannot provide a benefit, while still allowing stronger overlap heuristics in complex applications.
