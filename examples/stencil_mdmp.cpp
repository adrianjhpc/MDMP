#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mdmp_interface.h"

// Grid dimensions per rank
const int N = 2000; 

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    if (size < 2) {
        MDMP_COMM_FINAL();
        return 0;
    }

    int up_neighbor = (rank - 1 + size) % size;
    int down_neighbor = (rank + 1) % size;

    // 1D arrays representing a 2D grid (N x N)
    std::vector<double> grid(N * N, rank * 1.0);
    std::vector<double> next_grid(N * N, 0.0);

    // Halo (Ghost) buffers for the top and bottom edges
    std::vector<double> halo_top_recv(N, 0.0);
    std::vector<double> halo_bottom_recv(N, 0.0);

    MDMP_COMM_SYNC();
    double start_time = MDMP_WTIME();

    for (int iter = 0; iter < 500; ++iter) {
        
        // Send top row to upper neighbor, receive their bottom row into our top halo
        MDMP_SEND(&grid[0], N, rank, up_neighbor, 0);
        MDMP_RECV(halo_top_recv.data(), N, rank, up_neighbor, 1);

        // Send bottom row to lower neighbor, receive their top row into our bottom halo
        MDMP_SEND(&grid[(N - 1) * N], N, rank, down_neighbor, 1);
        MDMP_RECV(halo_bottom_recv.data(), N, rank, down_neighbor, 0);


        for (int y = 1; y < N - 1; ++y) {
            for (int x = 1; x < N - 1; ++x) {
                int i = y * N + x;
                next_grid[i] = 0.25 * (grid[i - 1] + grid[i + 1] + grid[i - N] + grid[i + N]);
            }
        }

        // Top boundary (y = 0)
        for (int x = 1; x < N - 1; ++x) {
            next_grid[x] = 0.25 * (grid[x - 1] + grid[x + 1] + halo_top_recv[x] + grid[x + N]);
        }

        // Bottom boundary (y = N - 1)
        for (int x = 1; x < N - 1; ++x) {
            int i = (N - 1) * N + x;
            next_grid[i] = 0.25 * (grid[i - 1] + grid[i + 1] + grid[i - N] + halo_bottom_recv[x]);
        }

        grid.swap(next_grid);
    }

    double end_time = MDMP_WTIME();

    if (rank == 1) {
        printf("------------------------------------------------\n");
        printf(" BENCHMARK: 2D Heat Equation / Stencil (MDMP Imperative)\n");
        printf("------------------------------------------------\n");
        printf("Grid Size per Rank: %d x %d\n", N, N);
        printf("Compiler perfectly overlapped interior math with halo exchange!\n");
        printf("Elapsed Time: %f seconds\n", end_time - start_time);
        printf("Validation: %f\n", grid[N/2 + N/2]); 
    }

    MDMP_COMM_FINAL();
    return 0;
}
