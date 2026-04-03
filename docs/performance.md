# Performance Notes

MDMP is designed to improve communication overlap and reduce communication overhead, but performance depends strongly on the structure of the application.

## When MDMP helps most

MDMP tends to help most when:

- communication can be initiated early
- data is not needed immediately
- there is enough useful computation to hide communication latency
- many small transfers can be batched or coalesced
- communication patterns repeat across timesteps or iterations

## When speedups may be small

Speedups may be limited when:

- message sizes are very small
- communication already costs little relative to computation
- the application is dominated by local arithmetic
- the communicated data is needed immediately after it is sent or received

## Imperative vs declarative performance

### Imperative mode
Usually best when:
- communication is explicit
- message count is modest
- overlap comes from issuing non-blocking point-to-point operations early

### Declarative mode
Usually best when:
- a communication phase contains many small or scattered transfers
- batching reduces request overhead
- repeated communication schedules benefit from runtime coalescing

## Practical advice

- benchmark against a hand-written non-blocking MPI version, not only blocking MPI
- use realistic multi-rank and multi-node runs
- test strong-scaling regimes where communication cost is significant
- inspect whether useful work exists between communication initiation and first use

## Interpretation

If MDMP shows little benefit on a kernel, it may mean:

- the communication was already effectively hidden
- the workload is compute-dominated
- the overlap window is too small
- batching overhead is larger than the communication savings
