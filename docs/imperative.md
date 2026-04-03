# Imperative MDMP

Imperative MDMP is the lower-level MDMP programming model. It is intended for applications that already have explicit communication logic and want the compiler to improve overlap automatically.

## Model

In imperative mode, the programmer writes communication operations directly:

- `MDMP_SEND`
- `MDMP_RECV`
- MDMP collectives such as:
  - `MDMP_REDUCE`
  - `MDMP_GATHER`
  - `MDMP_ALLREDUCE`
  - `MDMP_ALLGATHER`
  - `MDMP_BCAST`

The compiler pass lowers these to non-blocking runtime calls and inserts waits at the first safe point where completion is required.

## Example

```cpp
MDMP_RECV(halo_top_recv.data(), N, rank, up_neighbor, 1);
MDMP_SEND(&grid[0], N, rank, up_neighbor, 0);

MDMP_RECV(halo_bottom_recv.data(), N, rank, down_neighbor, 0);
MDMP_SEND(&grid[(N - 1) * N], N, rank, down_neighbor, 1);

// Interior computation here
```
This expresses the communication at a natural source-level location. MDMP then tries to:

 - start communication as early as possible
 - delay waiting as long as possible
 - overlap the in-flight communication with useful computation

## What the compiler does

For each imperative MDMP call, the compiler pass:

 - converts the marker into a runtime call such as `mdmp_send()` or `mdmp_recv()`
 - derives the communicated memory region
 - attempts to hoist the runtime call upward when there are no data hazards
 - tracks where the communicated data is first consumed or overwritten
 - inserts a wait only when needed

## Best use cases

Imperative MDMP works best when:

 - the communicated buffers are explicit
 - there is useful computation after the communication call
 - communication is needed later in the same iteration or region

## Caveats

Imperative MDMP cannot create overlap if:

 - the communicated data is needed immediately
 - there is no useful work between communication initiation and first use
 - message sizes are tiny and communication cost is already negligible


