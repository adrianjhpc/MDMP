#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <mpi.h>
#include "mdmp_pragma_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();
    int root = 0; // Define the root rank

    int send_val = rank + 10; // Ranks send: 10, 11, 12, 13...
    std::vector<int> recv_vals(size, 0);

    MDMP_GATHER(&send_val, 1, recv_vals.data(), root);

    bool success = true;
    
    if (rank == root) {
        for (int i = 0; i < size; ++i) {
            int expected = i + 10;
            if (recv_vals[i] != expected) {
                printf("Rank %d (ROOT): FAILED at index %d (Got %d, expected %d)\n", rank, i, recv_vals[i], expected);
                success = false;
            }
        }
        if (success) {
            printf("Rank %d (ROOT): SUCCESS (Gathered array perfectly)\n", rank);
        }
    } else {
        // Non-root ranks succeed automatically if they reach this point without deadlocking
        printf("Rank %d: SUCCESS (Sent data to root)\n", rank);
    }

    MDMP_COMM_FINAL();
    return success ? 0 : 1;
}
