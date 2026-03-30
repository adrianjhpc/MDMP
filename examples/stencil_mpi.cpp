#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <mpi.h>

const int N = 2000; 

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        MPI_Finalize();
        return 0;
    }

    int up_neighbor = (rank - 1 + size) % size;
    int down_neighbor = (rank + 1) % size;

    std::vector<double> grid(N * N, rank * 1.0);
    std::vector<double> next_grid(N * N, 0.0);
    std::vector<double> halo_top_recv(N, 0.0);
    std::vector<double> halo_bottom_recv(N, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int iter = 0; iter < 500; ++iter) {
        MPI_Request reqs[4];

        // Initiate non-blocking transfers
        MPI_Isend(&grid[0], N, MPI_DOUBLE, up_neighbor, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(halo_top_recv.data(), N, MPI_DOUBLE, up_neighbor, 1, MPI_COMM_WORLD, &reqs[1]);

        MPI_Isend(&grid[(N - 1) * N], N, MPI_DOUBLE, down_neighbor, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(halo_bottom_recv.data(), N, MPI_DOUBLE, down_neighbor, 0, MPI_COMM_WORLD, &reqs[3]);

        // Overlap Window: Compute the Interior
        for (int y = 1; y < N - 1; ++y) {
            for (int x = 1; x < N - 1; ++x) {
                int i = y * N + x;
                next_grid[i] = 0.25 * (grid[i - 1] + grid[i + 1] + grid[i - N] + grid[i + N]);
            }
        }

        // Manual Synchronization
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // Compute Boundaries
        for (int x = 1; x < N - 1; ++x) { // Top
            next_grid[x] = 0.25 * (grid[x - 1] + grid[x + 1] + halo_top_recv[x] + grid[x + N]);
        }
        for (int x = 1; x < N - 1; ++x) { // Bottom
            int i = (N - 1) * N + x;
            next_grid[i] = 0.25 * (grid[i - 1] + grid[i + 1] + grid[i - N] + halo_bottom_recv[x]);
        }

        grid.swap(next_grid);
    }

    double end_time = MPI_Wtime();

    if (rank == 1) {
        printf("------------------------------------------------\n");
        printf(" BENCHMARK: 2D Heat Equation (Raw MPI)\n");
        printf("------------------------------------------------\n");
        printf("Elapsed Time: %f seconds\n", end_time - start_time);
        printf("Validation: %f\n", grid[N/2 + N/2]); 
    }

    MPI_Finalize();
    return 0;
}
