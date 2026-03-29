#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mdmp_pragma_interface.h"

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

    const int num_migrating = 10000; // 10,000 particles cross the boundary
    const int iterations = 100;

    // Simulate our local particle lists
    std::vector<Particle> send_list(num_migrating);
    std::vector<Particle> recv_list(num_migrating);

    int right_neighbor = (rank + 1) % size;
    int left_neighbor = (rank - 1 + size) % size;

    MDMP_COMM_SYNC();
    double start_time = MDMP_WTIME();

    for (int iter = 0; iter < iterations; ++iter) {
        
        for (int i = 0; i < num_migrating; ++i) {
            // Count is 1 element! MDMP calculates the 56 bytes automatically.
            MDMP_SEND(&send_list[i], 1, rank, right_neighbor, 0);
            MDMP_RECV(&recv_list[i], 1, rank, left_neighbor, 0);
        }    

        double dummy_work = 0.0;
        for (int i = 0; i < 100000; ++i) {
            dummy_work += 0.0001; 
        }

        for (int i = 0; i < num_migrating; ++i) {
            send_list[i].x = recv_list[i].x + dummy_work; // Fake update
        }
    }

    double end_time = MDMP_WTIME();

    if (rank == 0) {
        printf("------------------------------------------------\n");
        printf(" BENCHMARK: N-Body Particle Exchange (Imperative)\n");
        printf("------------------------------------------------\n");
        printf("Particles Exchanged: %d per step\n", num_migrating);
        printf("Iterations: %d\n", iterations);
        printf("Elapsed Time: %f seconds\n", end_time - start_time);
    }

    MDMP_COMM_FINAL();
    return 0;
}
