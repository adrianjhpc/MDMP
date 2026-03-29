#include <iostream>
#include <vector>
#include <cstdlib>
#include "mdmp_interface.h"

int main() {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    // Setup local data (3 items per rank)
    int send_count = 3;
    double* send_buf = (double*)malloc(send_count * sizeof(double));

    for (int i = 0; i < send_count; ++i) {
        // e.g., Rank 2 gets [20.0, 21.0, 22.0]
        send_buf[i] = (double)(rank * 10 + i);
    }

    // Setup root receive buffer
    double* recv_buf = NULL;
    if (rank == 0) {
        // Root must allocate enough space for every rank's data
        recv_buf = (double*)malloc(size * send_count * sizeof(double));

        // Initialize with garbage to ensure we don't accidentally pass
        for (int i = 0; i < size * send_count; ++i) {
            recv_buf[i] = -1.0;
        }

        std::cout << "=== MDMP Declarative Gather Test ===" << std::endl;
        std::cout << "World Size: " << size << " | Items per rank: " << send_count << "\n" << std::endl;
    }

    // ========================================================================
    // INSPECTOR-EXECUTOR COMMUNICATION REGION
    // ========================================================================
    MDMP_COMMREGION_BEGIN();

    // PHASE 1: The "Inspector"
    // Every rank registers its contribution.
    // Only Rank 0 (the root) needs a valid recv_buf; others safely pass NULL.
    MDMP_REGISTER_GATHER(send_buf, send_count, recv_buf, 0);

    // PHASE 2: The "Executor"
    // The runtime dispatches the non-blocking MPI_Igather.
    MDMP_COMMIT();

    // PHASE 3: Computation
    // Process is free to do local work here while the network gathers the data.
    // The compiler mathematically proves it must sink the Waitall below this line!
    double local_work = send_buf[0] * 3.14;

    MDMP_COMMREGION_END();
    // ========================================================================

    // Automated verification (only root has the full gathered array)
    if (rank == 0) {
        bool passed = true;
        std::cout << "Gathered Array: [ ";

        for (int i = 0; i < size * send_count; ++i) {
            std::cout << recv_buf[i] << " ";

            // Calculate which rank this piece of data should have come from
            int source_rank = i / send_count;
            int offset = i % send_count;
            double expected = (double)(source_rank * 10 + offset);

            if (recv_buf[i] != expected) {
                passed = false;
            }
        }
        std::cout << "]\n" << std::endl;

        if (passed) {
            std::cout << "[PASS] Root successfully registered, gathered, and stitched the global array!" << std::endl;
        } else {
            std::cout << "[FAIL] Data mismatch detected in the gathered array!" << std::endl;
        }
    }

    // Cleanup
    if (rank == 0) free(recv_buf);
    free(send_buf);

    MDMP_COMM_FINAL();
    return 0;
}
