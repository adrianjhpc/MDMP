#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mdmp_pragma_interface.h"

// A custom 12-byte struct
struct Vector3 {
    float x, y, z;
};

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    if (size < 2) {
        printf("Test requires at least 2 ranks.\n");
        MDMP_COMM_FINAL();
        return 0;
    }

    // efine heterogeneous data types
    int my_int = 0;
    double my_doubles[3] = {0.0, 0.0, 0.0};
    char my_chars[2] = {'X', 'X'};
    Vector3 my_struct = {0.0f, 0.0f, 0.0f};

    // Rank 0 populates the data
    if (rank == 0) {
        my_int = 42;
        my_doubles[0] = 1.1; my_doubles[1] = 2.2; my_doubles[2] = 3.3;
        my_chars[0] = 'H'; my_chars[1] = 'I';
        my_struct.x = 10.0f; my_struct.y = 20.0f; my_struct.z = 30.0f;
    }

    MDMP_COMMREGION_BEGIN();

    if (rank == 0) {
        MDMP_REGISTER_SEND(&my_int,      1, rank, 1, 0); // Type 0 (Int)
        MDMP_REGISTER_SEND(my_doubles,   3, rank, 1, 0); // Type 1 (Double array)
        MDMP_REGISTER_SEND(my_chars,     2, rank, 1, 0); // Type 3 (Char array)
        MDMP_REGISTER_SEND(&my_struct,   1, rank, 1, 0); // Type 4 (Unknown/MPI_BYTE struct)
    } 
    else if (rank == 1) {
        MDMP_REGISTER_RECV(&my_int,      1, rank, 0, 0);
        MDMP_REGISTER_RECV(my_doubles,   3, rank, 0, 0);
        MDMP_REGISTER_RECV(my_chars,     2, rank, 0, 0);
        MDMP_REGISTER_RECV(&my_struct,   1, rank, 0, 0);
    }

    // Commits the heterogeneous bundle into ONE MPI_Isend
    MDMP_COMMIT(); 

    // Waits and unpacks
    MDMP_COMMREGION_END();

    // ====================================================================
    // VERIFICATION
    // ====================================================================
    bool success = true;
    if (rank == 1) {
        if (my_int != 42) success = false;
        if (fabs(my_doubles[1] - 2.2) > 0.001) success = false;
        if (my_chars[0] != 'H' || my_chars[1] != 'I') success = false;
        if (fabs(my_struct.z - 30.0f) > 0.001) success = false;

        if (!success) {
            printf("Rank 1 FAILED! Multi-type packing corrupted the data.\n");
            printf("Got Int: %d, Double[1]: %f, Chars: %c%c, Struct.z: %f\n", 
                   my_int, my_doubles[1], my_chars[0], my_chars[1], my_struct.z);
        } else {
            printf("Rank 1 SUCCESS! Packed an Int, Double Array, Char Array, and Custom Struct perfectly!\n");
        }
    }

    MDMP_COMM_FINAL();
    return success ? 0 : 1;
}
