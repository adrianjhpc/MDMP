#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include "mdmp_pragma_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    int send_val = rank + 1; // Ranks send: 1, 2, 3, 4...
    int recv_val = 0;

    // Execute Imperative Allreduce
    MDMP_ALLREDUCE(&send_val, &recv_val, 1, MPI_SUM);

    // The sum of 1 to N is N*(N+1)/2
    int expected_sum = (size * (size + 1)) / 2;
    
    if (recv_val == expected_sum) {
        printf("Rank %d: SUCCESS (Got expected sum: %d)\n", rank, recv_val);
    } else {
        printf("Rank %d: FAILED (Got %d, expected %d)\n", rank, recv_val, expected_sum);
        MDMP_COMM_FINAL();
        return 1;
    }

    MDMP_COMM_FINAL();
    return 0;
}
