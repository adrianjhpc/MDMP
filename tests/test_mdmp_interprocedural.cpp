// tests/test_mdmp_interprocedural.cpp
#include <iostream>
#include <vector>
#include <iomanip>

#include "mdmp_interface.h"

// ---------------------------------------------------------
// Wrapper Functions
// If compiling without -flto, uncomment the always_inline 
// attribute below to force the compiler to merge these into main().
// ---------------------------------------------------------
// #define FORCE_INLINE inline __attribute__((always_inline))
#define FORCE_INLINE 

FORCE_INLINE void do_mdmp_send(double* buf, int count, int actor, int peer, int tag) {
    MDMP_SEND(buf, count, actor, peer, tag);
}

FORCE_INLINE void do_mdmp_recv(double* buf, int count, int actor, int peer, int tag) {
    MDMP_RECV(buf, count, actor, peer, tag);
}

int main() {
    int mdmp_rank, mdmp_size;

    MDMP_COMM_INIT();

    mdmp_rank = MDMP_GET_RANK();
    mdmp_size = MDMP_GET_SIZE();

    if (mdmp_rank == 0) {
        std::cout << "=== Interprocedural MDMP Test ===" << std::endl;
        std::cout << "Testing Wait scheduling across function boundaries..." << std::endl;
    }

    const int size = 10000;
    std::vector<double> data(size, 0.0);
    std::vector<double> data2(size, 0.0);
    std::vector<double> result(size, 0.0);
    std::vector<double> result2(size, 0.0);
    double unrelated_work = 0.0;

    if(mdmp_rank < 2){

    // Initialize data on Rank 0
    if (mdmp_rank == 0) {
        for (int i = 0; i < size; ++i) {
            data[i] = static_cast<double>(i);
            data2[i] = static_cast<double>(i+20000);
        }
    }

    // Initiate network operations via the external wrapper functions
    do_mdmp_send(data.data(), size, 0, 1, 0);   // Rank 0 sends to Rank 1
    do_mdmp_send(data2.data(), size, 0, 1, 0);   // Rank 0 sends to Rank 1
    do_mdmp_recv(result.data(), size, 1, 0, 0); // Rank 1 recvs from Rank 0
    do_mdmp_recv(result2.data(), size, 1, 0, 0); // Rank 1 recvs from Rank 0
    do_mdmp_send(result.data(), size, 1, 0, 0); // Rank 1 sends to Rank 0
    do_mdmp_send(result2.data(), size, 1, 0, 0); // Rank 1 sends to Rank 0
    do_mdmp_recv(result.data(), size, 0, 1, 0); // Rank 0 recvs from Rank 1
    do_mdmp_recv(result2.data(), size, 0, 1, 0); // Rank 0 recvs from Rank 1

    // Unrelated compute
    // If the compiler pass successfully sees across the boundary (via inlining), 
    // it will not put the wait inside the wrappers. It will allow this 
    // compute to overlap with the network operations.
    for (int i = 0; i < size; ++i) {
        unrelated_work += (i * 0.001);
    }

    // Te compiler MemorySSA check should detect this overlapping store/load
    // and materialise the mdmp_wait() call right before this loop.
    for (int i = 0; i < size; ++i) {
        result[i] = result[i] * result[i] * result[i] + unrelated_work;
    }
    }

    MDMP_COMM_SYNC();
    MDMP_COMM_FINAL();

    if(mdmp_rank < 2){


    int errors = 0;
    for (int i = 0; i < size; ++i) {
        double expected = (static_cast<double>(i) * i * i) + unrelated_work;
        double expected2 = (static_cast<double>(i+20000) * i * i) + unrelated_work;
        // Floating point comparison with a small epsilon
        if (std::abs(result[i] - expected) > 1e-5) {
            errors++;
            // Only print the first few errors to avoid flooding the console
            if (errors <= 5) {
                std::cerr << "[Rank " << mdmp_rank << "] Mismatch at index " << i 
                          << " | Expected: " << expected 
                          << " | Got: " << result[i] << std::endl;
            }
        }
        if (std::abs(result2[i] - expected2) > 1e-5) {
            errors++;
            // Only print the first few errors to avoid flooding the console
            if (errors <= 5) {
                std::cerr << "[Rank " << mdmp_rank << "] Mismatch at index " << i
                          << " | Expected 2: " << expected2
                          << " | Got: " << result2[i] << std::endl;
            }
        }

    }

    if (mdmp_rank == 0) {
        if (errors == 0) {
            std::cout << "Validation: PASSED" << std::endl;
            std::cout << "First few correct results:" << std::endl;
            for(int i = 0; i < 3; i++) {
                std::cout << "  result[" << i << "] = " << std::fixed << std::setprecision(3) << result[i] << std::endl;
            }
            std::cout << "Interprocedural MDMP test completed successfully!" << std::endl;
        } else {
            std::cerr << "Validation: FAILED with " << errors << " errors on Rank 0." << std::endl;
            return 1; // Return non-zero to fail the test runner
        }
    } else {
        // If other ranks failed, we still want to exit with an error code
        if (errors > 0) {
            return 1;
        }
    }
    }
    return 0;
}
