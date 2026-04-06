#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include "mdmp_interface.h"

struct Particle {
    double x, y, z;
    double vx, vy, vz;
    int id;
    int type;
};

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    const int num_migrating = 10000; 
    const int iterations = 100;

    std::vector<Particle> send_list(num_migrating);
    std::vector<Particle> recv_list(num_migrating);

    // SEED INITIAL STATE: Set z to the current rank ID
    for (int i = 0; i < num_migrating; ++i) {
        send_list[i].z = (double)rank;
    }

    int right_neighbour = (rank + 1) % size;
    int left_neighbour = (rank - 1 + size) % size;

    MDMP_COMM_SYNC();
    double start_time = MDMP_WTIME();

    for (int iter = 0; iter < iterations; ++iter) {
        
        MDMP_COMMREGION_BEGIN();
        for (int i = 0; i < num_migrating; ++i) {
            MDMP_REGISTER_SEND(&send_list[i], 1, rank, right_neighbour, 0);
            MDMP_REGISTER_RECV(&recv_list[i], 1, rank, left_neighbour, 0);
        }
        MDMP_COMMIT();

        double dummy_work = 0.0;
        for (int i = 0; i < 100000; ++i) {
            dummy_work += 0.0001; 
        }

        MDMP_COMMREGION_END();

        for (int i = 0; i < num_migrating; ++i) {
            send_list[i].z = recv_list[i].z + dummy_work; 
        }
    }

    double end_time = MDMP_WTIME();

    // ==========================================
    // CORRECTNESS CHECK
    // ==========================================
    
    // Calculate which rank's data we should be holding after 'iterations' shifts
    int expected_origin_rank = (rank - (iterations % size) + size) % size;

    // Calculate the exact expected floating-point accumulation
    double expected_dummy_total = 0.0;
    for (int iter = 0; iter < iterations; ++iter) {
        double dummy_work = 0.0;
        for (int i = 0; i < 100000; ++i) { dummy_work += 0.0001; }
        expected_dummy_total += dummy_work;
    }

    double expected_z = (double)expected_origin_rank + expected_dummy_total;
   
 
    bool correct = true;
    for (int i = 0; i < num_migrating; ++i) {
        // Use an epsilon to account for standard IEEE 754 floating-point jitter
        if (std::abs(send_list[i].z - expected_z) > 1e-5) {
            correct = false;
            break;
        }
    }

    if (!correct) {
        printf("[MDMP] Rank %d Validation FAILED! Expected z = %f, but got %f\n", rank, expected_z, send_list[0].z);
    } else {
        if (rank == 0) {
            printf("[MDMP] Rank %d Validation PASSED!\n", rank);
        }
    }
    // ==========================================

    if (rank == 0) {
        printf("------------------------------------------------\n");
        printf(" BENCHMARK: N-Body Particle Exchange (Declarative)\n");
        printf("------------------------------------------------\n");
        printf("Running on %d processes\n", size);
        printf("Particles Exchanged: %d per step\n", num_migrating);
        printf("Iterations: %d\n", iterations);
        printf("Elapsed Time: %f seconds\n", end_time - start_time);
    }

    MDMP_COMM_FINAL();
    return 0;
}
