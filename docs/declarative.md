# Declarative MDMP

Declarative MDMP is designed for communication patterns that are easier to describe as a batch than as a sequence of individual point-to-point operations.

## Model

A communication region is written as:

1. begin region
2. register communication operations
3. commit the batch
4. end region

Typical primitives include:

- `MDMP_COMM_REGION_BEGIN()`
- `MDMP_REGISTER_SEND(...)`
- `MDMP_REGISTER_RECV(...)`
- `MDMP_REGISTER_REDUCE(...)`
- `MDMP_REGISTER_GATHER(...)`
- `MDMP_REGISTER_ALLREDUCE(...)`
- `MDMP_REGISTER_ALLGATHER(...)`
- `MDMP_REGISTER_BCAST(...)`
- `MDMP_COMMIT()`
- `MDMP_COMM_REGION_END()`

## Example

```cpp
MDMP_COMM_REGION_BEGIN();

MDMP_REGISTER_RECV(recvbuf.data(), 1024, rank, src, 0);
MDMP_REGISTER_SEND(sendbuf.data(), 1024, rank, dest, 0);

MDMP_COMMIT();

MDMP_COMM_REGION_END();
```

## Runtime behavior

At commit time, MDMP may:

 - sort communication operations by peer and tag
 - combine multiple registered operations
 - build derived MPI datatypes for scattered memory regions
 - launch non-blocking communication as one declarative batch

Declarative mode is especially useful when many small communication fragments should be treated as a single communication schedule.

## What the compiler does

The compiler pass treats a committed declarative region as an asynchronous communication batch. It tracks the memory regions involved and inserts completion waits before the first true use or overwrite of those regions.

## Best use cases

Declarative MDMP is especially useful when:

 - many small communication operations belong to one logical phase
 - data is naturally scattered in memory
 - communication can be registered first and launched together
 - batching reduces request or datatype overhead

## Notes

Declarative mode is most effective when the same communication pattern repeats over many iterations, because batching and coalescing amortize overhead.


