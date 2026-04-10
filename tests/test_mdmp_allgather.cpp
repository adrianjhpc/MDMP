#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "mdmp_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    int send_val = rank + 10; // Ranks send: 10, 11, 12, 13...
    std::vector<int> recv_vals(size, 0);

    // Execute Imperative Allgather (No root argument needed)
    MDMP_ALLGATHER(&send_val, 1, recv_vals.data());

    bool success = true;
    
    // EVERY rank checks the final array
    for (int i = 0; i < size; ++i) {
        int expected = i + 10;
        if (recv_vals[i] != expected) {
            printf("Rank %d: FAILED at index %d (Got %d, expected %d)\n", rank, i, recv_vals[i], expected);
            success = false;
        }
    }

    if (success) {
        printf("Rank %d: SUCCESS (Gathered array perfectly)\n", rank);
    }

    MDMP_COMM_FINAL();
    return success ? 0 : 1;
}
