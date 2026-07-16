// Regression test for the "read-only send buffer never drained" liveness bug.
//
// Background:
//   The MDMP pass only emits an explicit wait for a buffer at its first
//   conflicting access. A send buffer that is never rewritten inside a comm
//   region therefore gets NO explicit wait. When MPI is initialized without
//   MPI_THREAD_MULTIPLE (no background progress thread), such sends only make
//   progress while the rank is inside an MPI call. If ranks desynchronize and
//   enter compute, these un-waited sends stall, and their matching receives on
//   peers hang -> global deadlock.
//
//   Historically mdmp_wait() drained ALL in-flight imperative requests via a
//   single MPI_Waitall, which incidentally completed these read-only sends.
//   If that draining behavior is ever removed by default in the no-thread
//   configuration, this test deadlocks and the CTest TIMEOUT flags it.
//
// This test MUST complete within its CTest timeout. A hang == regression.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "mdmp_interface.h"

int main() {
  MDMP_COMM_INIT();

  const int rank = MDMP_GET_RANK();
  const int size = MDMP_GET_SIZE();

  if (size < 2) {
    if (rank == 0)
      std::fprintf(stderr, "[TEST] needs >= 2 ranks\n");
    MDMP_COMM_FINAL();
    return 1;
  }

  // Ring neighbours.
  const int right = (rank + 1) % size;
  const int left  = (rank - 1 + size) % size;

  // Large enough to force the rendezvous protocol: eager completion could
  // mask the bug by letting the send finish without the peer progressing.
  const int N = 200000;           // 200k doubles = ~1.6 MB per message
  const int TAG = 42;
  const int ITERS = 100;          // many rounds -> desync is near-certain

  // Read-only send buffer: filled ONCE, never modified inside a region.
  // This is what prevents the pass from generating a wait for the send.
  std::vector<double> sendbuf(N, static_cast<double>(rank) + 1.0);
  std::vector<double> recvbuf(N, -1.0);

  double checksum = 0.0;
  int errors = 0;

  for (int it = 0; it < ITERS; ++it) {
    MDMP_COMMREGION_BEGIN();

    // Post recv from left, send (read-only buf) to right.
    MDMP_RECV(recvbuf.data(), N, rank, left,  TAG);
    MDMP_SEND(sendbuf.data(), N, rank, right, TAG);

    MDMP_COMMREGION_END();

    // Reading recvbuf forces the recv wait to materialise here.
    // We deliberately DO NOT touch sendbuf, so the send has no wait.
    const double expected = static_cast<double>(left) + 1.0;
    if (recvbuf[0] != expected || recvbuf[N - 1] != expected)
      ++errors;
    checksum += recvbuf[N / 2];

    // Deliberate, rank-proportional load imbalance to desynchronise ranks so
    // some are deep in compute while neighbours are still waiting -> exactly
    // the condition under which an un-drained send deadlocks.
    volatile double spin = 0.0;
    const long work = static_cast<long>(rank + 1) * 2000000L;
    for (long i = 0; i < work; ++i)
      spin += 1.0e-9;
    checksum += spin * 0.0; // keep 'spin' live, contribute nothing
  }

  // Reduce error count so any rank's failure fails the whole test.
  int global_errors = 0;
  MDMP_COMMREGION_BEGIN();
  MDMP_ALLREDUCE(&errors, &global_errors, 1, MDMP_SUM);
  MDMP_COMMREGION_END();
  // Touch the reduce output to force its completion.
  volatile int sink = global_errors;
  (void)sink;

  if (rank == 0) {
    if (global_errors == 0)
      std::printf("[TEST] readonly_send_drain PASSED (checksum=%f)\n", checksum);
    else
      std::printf("[TEST] readonly_send_drain FAILED: %d data errors\n",
                  global_errors);
  }

  MDMP_COMM_FINAL();
  return global_errors == 0 ? 0 : 1;
}

