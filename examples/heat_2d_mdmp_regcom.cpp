#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mdmp_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    // 1D Domain Decomposition (Row-wise slice)
    const int Nx = 2000;         
    const int local_Ny = 2000;   
    const int iterations = 1000;

    std::vector<double> grid((local_Ny + 2) * Nx, 0.0);
    std::vector<double> grid_new((local_Ny + 2) * Nx, 0.0);

    // Heat source
    if (rank == size / 2) {
        grid[(local_Ny / 2 + 1) * Nx + (Nx / 2)] = 1000.0;
    }

    double* top_halo_recv = &grid[0];                     
    double* top_halo_send = &grid[1 * Nx];                
    double* bot_halo_send = &grid[local_Ny * Nx];         
    double* bot_halo_recv = &grid[(local_Ny + 1) * Nx];   

    MDMP_COMM_SYNC();
    double start_time = MDMP_WTIME();

    for (int iter = 0; iter < iterations; ++iter) {
        
        MDMP_COMMREGION_BEGIN();
        
        if (rank > 0) {
            MDMP_REGISTER_SEND(top_halo_send, Nx, rank, rank - 1, 0); 
            MDMP_REGISTER_RECV(top_halo_recv, Nx, rank, rank - 1, 1); 
        }
        if (rank < size - 1) {
            MDMP_REGISTER_SEND(bot_halo_send, Nx, rank, rank + 1, 1); 
            MDMP_REGISTER_RECV(bot_halo_recv, Nx, rank, rank + 1, 0); 
        }
        
        MDMP_COMMIT();

        for (int y = 2; y <= local_Ny - 1; ++y) {
            for (int x = 1; x < Nx - 1; ++x) {
                grid_new[y * Nx + x] = 0.25 * (
                    grid[(y - 1) * Nx + x] + grid[(y + 1) * Nx + x] +
                    grid[y * Nx + (x - 1)] + grid[y * Nx + (x + 1)]
                );
            }
        }

        MDMP_COMMREGION_END();

        if (rank > 0) {
            int y = 1; 
            for (int x = 1; x < Nx - 1; ++x) {
                grid_new[y * Nx + x] = 0.25 * (
                    grid[(y - 1) * Nx + x] + grid[(y + 1) * Nx + x] +
                    grid[y * Nx + (x - 1)] + grid[y * Nx + (x + 1)]
                );
            }
        }

        if (rank < size - 1) {
            int y = local_Ny; 
            for (int x = 1; x < Nx - 1; ++x) {
                grid_new[y * Nx + x] = 0.25 * (
                    grid[(y - 1) * Nx + x] + grid[(y + 1) * Nx + x] +
                    grid[y * Nx + (x - 1)] + grid[y * Nx + (x + 1)]
                );
            }
        }

        grid.swap(grid_new);
    }

    double end_time = MDMP_WTIME();

    double calc_time = (end_time - start_time);
    double max_time = 0.0;
    MDMP_REDUCE(&calc_time, &max_time, 1, 0, MDMP_MAX);
    
    if (rank == 0) {
        printf("MDMP (RegCom) 2D Heat Equation Benchmark\n");
        printf("Grid Size: %d x %d per rank\n", Nx, local_Ny);
        printf("Iterations: %d\n", iterations);
        printf("Elapsed Time: %f seconds\n", max_time);
    }

    MDMP_COMM_FINAL();
    return 0;
}
