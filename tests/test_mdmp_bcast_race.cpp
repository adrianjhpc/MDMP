#include <stdio.h>
#include <stdlib.h>
#include "mdmp_interface.h"

int main(int argc, char** argv) {
    // 1. Initialize Runtime
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    // 2. Setup the target variable
    // Ranks 1+ initialize this to 0. Rank 0 initializes to the target 42.
    int expected_value = 42;
    int broadcast_value = 0; 

    if (rank == 0) {
        broadcast_value = expected_value;
    }

    // ---------------------------------------------------------
    // 3. The Critical Communication Region
    // ---------------------------------------------------------
    // If these boundary macros are removed, the asynchronous network 
    // card will race the CPU to the if-statement below!
    
    MDMP_BCAST(&broadcast_value, 1, 0);
    
    // ---------------------------------------------------------
    // 4. Immediate CPU Evaluation
    // ---------------------------------------------------------
    int errors = 0;
    
    if (broadcast_value == expected_value) {
        printf("[SUCCESS] Rank %d correctly read synchronized memory: %d\n", rank, broadcast_value);
    } else {
        printf("[FAILURE] Rank %d experienced a ghost read! Expected %d but got %d\n", 
               rank, expected_value, broadcast_value);
        errors++;
    }

    // 5. Clean Shutdown
    MDMP_COMM_FINAL();
    
    // CTest relies on the exit code. 0 = Pass, >0 = Fail.
    return errors > 0 ? 1 : 0;

}
