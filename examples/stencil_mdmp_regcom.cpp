#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mdmp_interface.h"

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

    std::vector<double> grid(N * N, rank * 1.0);
    std::vector<double> next_grid(N * N, 0.0);
    std::vector<double> halo_top_recv(N, 0.0);
    std::vector<double> halo_bottom_recv(N, 0.0);

    MDMP_COMM_SYNC();
    double start_time = MDMP_WTIME();

    for (int iter = 0; iter < 500; ++iter) {
        
        MDMP_COMMREGION_BEGIN();
        
        // Register top and bottom edge exchanges
        MDMP_REGISTER_SEND(&grid[0], N, rank, up_neighbor, 0);
        MDMP_REGISTER_RECV(halo_top_recv.data(), N, rank, up_neighbor, 1);

        MDMP_REGISTER_SEND(&grid[(N - 1) * N], N, rank, down_neighbor, 1);
        MDMP_REGISTER_RECV(halo_bottom_recv.data(), N, rank, down_neighbor, 0);

        // Push all registered requests to the hardware
        MDMP_COMMIT(); 

        // Because this loop is inside the region, it overlaps with the Commit
        for (int y = 1; y < N - 1; ++y) {
            for (int x = 1; x < N - 1; ++x) {
                int i = y * N + x;
                next_grid[i] = 0.25 * (grid[i - 1] + grid[i + 1] + grid[i - N] + grid[i + N]);
            }
        }

        MDMP_COMMREGION_END();

        for (int x = 1; x < N - 1; ++x) { // Top
            next_grid[x] = 0.25 * (grid[x - 1] + grid[x + 1] + halo_top_recv[x] + grid[x + N]);
        }
        for (int x = 1; x < N - 1; ++x) { // Bottom
            int i = (N - 1) * N + x;
            next_grid[i] = 0.25 * (grid[i - 1] + grid[i + 1] + grid[i - N] + halo_bottom_recv[x]);
        }

        grid.swap(next_grid);
    }

    double end_time = MDMP_WTIME();

    double calc_time = (end_time - start_time);
    double max_time = 0.0;
    MDMP_REDUCE(&calc_time, &max_time, 1, 1, MDMP_MAX);
        
    if (rank == 1) {
        printf("------------------------------------------------\n");
        printf(" BENCHMARK: 2D Heat Equation (MDMP Declarative)\n");
        printf("------------------------------------------------\n");
        printf("Grid Size per Rank: %d x %d\n", N, N);
        printf("Elapsed Time: %f seconds\n", max_time);
        printf("Validation: %f\n", grid[N/2 + N/2]); 
    }

    MDMP_COMM_FINAL();
    return 0;
}
