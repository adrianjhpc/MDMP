#include <stdio.h>
#include <stdlib.h>
#include "mdmp_pragma_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    if (size < 2) {
        printf("Test requires at least 2 ranks.\n");
        MDMP_COMM_FINAL();
        return 0;
    }

    // Three completely separate variables scattered in memory
    double particle_x = 0.0;
    double particle_y = 0.0;
    int particle_id = 0;

    // Rank 0 populates the data
    if (rank == 0) {
        particle_x = 105.5;
        particle_y = 210.2;
        particle_id = 42;
    }

    MDMP_COMMREGION_BEGIN();

    if (rank == 0) {
        // Rank 0 registers 3 separate sends to Rank 1
        MDMP_REGISTER_SEND(&particle_x,  1, 0, 1, 0);
        MDMP_REGISTER_SEND(&particle_y,  1, 0, 1, 0);
        MDMP_REGISTER_SEND(&particle_id, 1, 0, 1, 0);
    } 
    else if (rank == 1) {
        // Rank 1 registers 3 separate recvs from Rank 0
        MDMP_REGISTER_RECV(&particle_x,  1, 1, 0, 0);
        MDMP_REGISTER_RECV(&particle_y,  1, 1, 0, 0);
        MDMP_REGISTER_RECV(&particle_id, 1, 1, 0, 0);
    }

    // Under the hood, this will malloc a 20-byte buffer (8+8+4), 
    // pack all three variables, and fire exactly 1 MPI_Isend.
    MDMP_COMMIT(); 

    // Under the hood, this will wait for the 1 MPI_Irecv, 
    // and unpack the 20-byte buffer back into the 3 separate pointers.
    MDMP_COMMREGION_END();

    // ==========================================
    // VERIFICATION
    // ==========================================
    bool success = true;
    if (rank == 1) {
        if (particle_x != 105.5 || particle_y != 210.2 || particle_id != 42) {
            printf("Rank 1 FAILED! Unpacked data mismatch. Got X:%.1f Y:%.1f ID:%d\n", 
                   particle_x, particle_y, particle_id);
            success = false;
        } else {
            printf("Rank 1 SUCCESS! Coalesced 3 scattered variables perfectly. X:%.1f Y:%.1f ID:%d\n", 
                   particle_x, particle_y, particle_id);
        }
    }

    MDMP_COMM_FINAL();
    return success ? 0 : 1;
}
