// tests/test_mdmp_deadlock.cpp
#include <iostream>
#include <vector>
#include <iomanip>

#include "mdmp_interface.h"

void do_mdmp_send(double* buf, int count, int actor, int peer, int tag);
void do_mdmp_recv(double* buf, int count, int actor, int peer, int tag);


int main() {
    int mdmp_rank, mdmp_size;

    MDMP_COMM_INIT();
    mdmp_rank = MDMP_GET_RANK();
    mdmp_size = MDMP_GET_SIZE();

    if (mdmp_rank == 0) {
        std::cout << "=== Interprocedural Deadlock Test ===" << std::endl;
    }

    const int size = 10000;
    std::vector<double> data(size, 1.0);
    std::vector<double> result(size, 0.0);

    if (mdmp_rank < 2) {
        // ---------------------------------------------------------
        // THE DEADLOCK ZONE
        // ---------------------------------------------------------
        if (mdmp_rank == 0) {
            do_mdmp_recv(result.data(), size, 0, 1, 0); 
            do_mdmp_send(data.data(), size, 0, 1, 0);   
        } 
        else if (mdmp_rank == 1) {
            do_mdmp_recv(result.data(), size, 1, 0, 0); 
            do_mdmp_send(data.data(), size, 1, 0, 0);   
        }

        // Buffer Consumption (The pass should hoist the wait to here)
        for (int i = 0; i < size; ++i) {
            result[i] = result[i] * 2.0;
        }
    }

    MDMP_COMM_SYNC();
    MDMP_COMM_FINAL();

    if (mdmp_rank == 0) {
        std::cout << "Test completed without deadlocking!" << std::endl;
    }

    return 0;
}
