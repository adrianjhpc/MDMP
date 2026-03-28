#include <stdio.h>
#include <stdlib.h>
#include "mdmp_pragma_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();

    // Give the runtime a second to initialize
    if (rank == 0) {
        printf("Rank 0: Initiating MDMP_ABORT with error code 42...\n");
    }

    // Trigger the forceful termination
    MDMP_ABORT(42);

    // --- EVERYTHING BELOW THIS LINE SHOULD BE DEAD CODE ---

    printf("Rank %d: FAILED! The program survived the abort!\n", rank);

    MDMP_COMM_FINAL();
    
    // If the program reaches this return 0, the abort failed, 
    // so returning 0 will actually cause CTest to fail the test 
    // (because CTest is expecting a crash).
    return 0; 
}
