# Runtime behavior

MDMP uses MPI underneath and maintains its own communication runtime.

## Initialization behavior

When `MDMP_COMM_INIT()` is called:

- if MPI is not yet initialized, MDMP initializes MPI
- if MPI is already initialized, MDMP attaches to the existing environment
- MDMP duplicates `MPI_COMM_WORLD` into its own internal communicator

## Rank and size

Use:

- `MDMP_GET_RANK()`
- `MDMP_GET_SIZE()`

to query the active MDMP/MPI environment.

## Timing and synchronization

MDMP provides:

- `MDMP_WTIME()`
- `MDMP_COMM_SYNC()`

which map to MPI-based timing and barriers.

## Environment variables

### `MDMP_DEBUG`
Enable MDMP runtime debug output.

Example:

```bash
MDMP_DEBUG=1 mpirun -n 4 ./my_mdmp_program
```
## Progress behavior

MDMP may use:

 - background progress when supported by the MPI environment
 - compiler-injected periodic progress calls in long-running loops

This depends on the runtime configuration and the MPI threading level available at initialization.


