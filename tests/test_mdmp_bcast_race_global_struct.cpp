#include <stdio.h>
#include <stdlib.h>
#include "mdmp_interface.h"

// Define a global struct 
struct SimulationParameters {
    int num_files;
    double box_size;
    int flags[4];
};

struct SimulationParameters GlobalHeader;

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    // Setup the split state
    // Rank 0 sets the correct data. Other ranks set it to 0.
    if (rank == 0) {
        GlobalHeader.num_files = 42;
        GlobalHeader.box_size = 100.0;
        GlobalHeader.flags[0] = 1;
    } else {
        GlobalHeader.num_files = 0;
        GlobalHeader.box_size = 0.0;
        GlobalHeader.flags[0] = 0;
    }

    // ---------------------------------------------------------
    // The Asynchronous Imperative Broadcast
    // ---------------------------------------------------------
    // We are deliberately not using COMMREGION boundaries here.
    // The LLVM pass must automatically detect the upcoming hazard 
    // and inject the Wait instruction.
    MDMP_BCAST(&GlobalHeader, 1, 0);

    // ---------------------------------------------------------
    // The GEP Hazard (GetElementPtr)
    // ---------------------------------------------------------
    // At the LLVM IR level, this reads from an offset of GlobalHeader.
    // The patched pass will see that getUnderlyingObject() matches 
    // the base pointer passed to MDMP_BCAST and inject a Wait right here.
    int errors = 0;
    
    if (GlobalHeader.num_files == 42) {
        printf("[SUCCESS] Rank %d safely read the synchronized struct field!\n", rank);
    } else {
        printf("[FAILURE] Rank %d suffered a ghost read! Expected 42 but got %d\n", 
               rank, GlobalHeader.num_files);
        errors++;
    }

    MDMP_COMM_FINAL();
    
    // CTest relies on the exit code. 0 = Pass, >0 = Fail.
    return errors > 0 ? 1 : 0;
}
