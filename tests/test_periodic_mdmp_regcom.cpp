#include <iostream>
#include <vector>
#include "mdmp_pragma_interface.h"

int main() {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    // Calculate the periodic boundaries
    // Adding 'size' before modulo prevents negative numbers in C++
    int left_neighbor  = (rank - 1 + size) % size;
    int right_neighbor = (rank + 1) % size;

    // Setup our tiny 3-element array: [Left Ghost, Local Data, Right Ghost]
    double left_ghost  = -1.0;
    double my_data     = (double)rank; // E.g., Rank 2 holds the value '2.0'
    double right_ghost = -1.0;

    if (rank == 0) {
        std::cout << "=== MDMP Declarative Periodic Boundary Test (Ring Topology) ===" << std::endl;
        std::cout << "World Size: " << size << std::endl;
    }

    MDMP_COMMREGION_BEGIN();

    // Tag 1: Register send left, receive from right
    MDMP_REGISTER_SEND(&my_data,      1, rank, left_neighbor,  1);
    MDMP_REGISTER_RECV(&right_ghost,  1, rank, right_neighbor, 1);

    // Tag 2: Register send right, receive from left
    MDMP_REGISTER_SEND(&my_data,      1, rank, right_neighbor, 2);
    MDMP_REGISTER_RECV(&left_ghost,   1, rank, left_neighbor,  2);

    // Dispatches all 4 boundary exchanges to the NIC simultaneously!
    MDMP_COMMIT();

    // PHASE 3: Computation
    // Simulate some local work while data is in flight.
    // The compiler mathematically proves it must sink the Waitall below this line.
    double local_computation = my_data * 3.14159;

    MDMP_COMMREGION_END();
    
    MDMP_COMM_SYNC();

    // Since we cast exact small integers to double, direct equality checking is safe here
    bool passed = true;
    if (left_ghost != (double)left_neighbor) passed = false;
    if (right_ghost != (double)right_neighbor) passed = false;

    // Print Results (Using a small barrier to keep prints relatively ordered)
    int failed = 0;
    for (int i = 0; i < size; ++i) {
        if (rank == i) {
            if(!passed){
                std::cout << "[FAIL] Rank " << rank << " | Expected Left: " << left_neighbor
                          << ", Got: " << left_ghost << " | Expected Right: " << right_neighbor
                          << ", Got: " << right_ghost << std::endl;
                failed = 1;
            }
        }
        MDMP_COMM_SYNC(); // Force ordering for neat terminal output
    }

    int failed_out = 0;
    MDMP_COMMREGION_BEGIN();
    MDMP_REGISTER_REDUCE(&failed, &failed_out, 1, 0, MDMP_SUM);
    MDMP_COMMIT();
    MDMP_COMMREGION_END();

    if (rank == 0) {
        if(failed_out == 0){
            std::cout << "[PASS] All periodic boundaries successfully exchanged!" << std::endl;
        }
    }

    MDMP_COMM_FINAL();
    return 0;
}
