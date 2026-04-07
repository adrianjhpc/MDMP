#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <mpi.h>

struct Particle {
    double x, y, z;
    double vx, vy, vz;
    int id;
    int type;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int num_migrating = 10000; // 10,000 particles cross the boundary
    const int iterations = 100;

    std::vector<Particle> send_list(num_migrating);
    std::vector<Particle> recv_list(num_migrating);

    int right_neighbour = (rank + 1) % size;
    int left_neighbour = (rank - 1 + size) % size;

    for (int i = 0; i < num_migrating; ++i) {
        send_list[i].z = (double)rank;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        
        std::vector<MPI_Request> reqs(num_migrating * 2);
        int req_idx = 0;

        for (int i = 0; i < num_migrating; ++i) {
            MPI_Isend(&send_list[i], sizeof(Particle), MPI_BYTE, right_neighbour, 0, MPI_COMM_WORLD, &reqs[req_idx++]);
            MPI_Irecv(&recv_list[i], sizeof(Particle), MPI_BYTE, left_neighbour, 0, MPI_COMM_WORLD, &reqs[req_idx++]);
        }

        double dummy_work = 0.0;
        for (int i = 0; i < 100000; ++i) {
            dummy_work += 0.0001; 
        }

        MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

        for (int i = 0; i < num_migrating; ++i) {
            send_list[i].z = recv_list[i].z + dummy_work; 
        }
    }
 
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

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

    double calc_time = (end_time - start_time);
    double max_time = 0.0;
    MPI_Reduce(&calc_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("------------------------------------------------\n");
        printf(" BENCHMARK: N-Body Particle Exchange (Raw MPI)\n");
        printf("------------------------------------------------\n");
        printf("Running on %d processes\n", size);
        printf("Particles Exchanged: %d per step\n", num_migrating);
        printf("Iterations: %d\n", iterations);
        printf("Elapsed Time: %f seconds\n", max_time);
    }

    MPI_Finalize();
    return 0;
}
