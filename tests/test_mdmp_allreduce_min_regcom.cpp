#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include "mdmp_pragma_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    int send_val = rank + 1;
    int recv_val = 0;

    // Execute Declarative Allreduce
    MDMP_COMMREGION_BEGIN();
    MDMP_REGISTER_ALLREDUCE(&send_val, &recv_val, 1, MPI_MIN);
    MDMP_COMMIT();
    MDMP_COMMREGION_END();

    int expected_value = 1;
    
    if (recv_val == expected_value) {
        printf("Rank %d: SUCCESS RegCom (Got expected value: %d)\n", rank, recv_val);
    } else {
        printf("Rank %d: FAILED RegCom (Got %d, expected %d)\n", rank, recv_val, expected_value);
        MDMP_COMM_FINAL();
        return 1;
    }

    MDMP_COMM_FINAL();
    return 0;
}
