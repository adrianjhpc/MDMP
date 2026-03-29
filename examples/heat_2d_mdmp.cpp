#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mdmp_pragma_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    // 1D Domain Decomposition (Row-wise slice)
    const int Nx = 2000;         // Grid width
    const int local_Ny = 2000;   // Local grid height per rank
    const int iterations = 1000;

    // Grids: Including +2 for top/bottom ghost rows (halos)
    std::vector<double> grid((local_Ny + 2) * Nx, 0.0);
    std::vector<double> grid_new((local_Ny + 2) * Nx, 0.0);

    // Heat source in the middle of the global domain
    if (rank == size / 2) {
        grid[(local_Ny / 2 + 1) * Nx + (Nx / 2)] = 1000.0;
    }

    // Pointers to the Halos
    double* top_halo_recv = &grid[0];                     // Row 0
    double* top_halo_send = &grid[1 * Nx];                // Row 1
    double* bot_halo_send = &grid[local_Ny * Nx];         // Row local_Ny
    double* bot_halo_recv = &grid[(local_Ny + 1) * Nx];   // Row local_Ny + 1

    MDMP_COMM_SYNC();
    double start_time = MDMP_WTIME();

    for (int iter = 0; iter < iterations; ++iter) {
        
        // ====================================================================
        // PHASE 1: NAIVE COMMUNICATION (No Requests, No Waits!)
        // In standard MPI, users often use blocking MPI_Sendrecv here, killing overlap.
        // MDMP non-blocking calls fire immediately into the background.
        // ====================================================================
        if (rank > 0) {
            MDMP_SEND(top_halo_send, Nx, rank, rank - 1, 0); // Send up
            MDMP_RECV(top_halo_recv, Nx, rank, rank - 1, 1); // Recv from up
        }
        if (rank < size - 1) {
            MDMP_SEND(bot_halo_send, Nx, rank, rank + 1, 1); // Send down
            MDMP_RECV(bot_halo_recv, Nx, rank, rank + 1, 0); // Recv from down
        }

        // ====================================================================
        // PHASE 2: INNER COMPUTE (Overlaps perfectly with network!)
        // The LLVM pass sees this loop DOES NOT touch top_halo_recv or bot_halo_recv.
        // It lets this loop run while the network hardware transfers the halos.
        // ====================================================================
        for (int y = 2; y <= local_Ny - 1; ++y) {
            for (int x = 1; x < Nx - 1; ++x) {
                grid_new[y * Nx + x] = 0.25 * (
                    grid[(y - 1) * Nx + x] + grid[(y + 1) * Nx + x] +
                    grid[y * Nx + (x - 1)] + grid[y * Nx + (x + 1)]
                );
            }
        }

        // ====================================================================
        // PHASE 3: BOUNDARY COMPUTE (The LLVM Wait Injection Point)
        // The LLVM pass traces the halo pointers here. It automatically injects
        // `mdmp_wait()` exactly right before these loops execute!
        // ====================================================================
        
        // Top Boundary
        if (rank > 0) {
            int y = 1; 
            for (int x = 1; x < Nx - 1; ++x) {
                // LLVM intercepts this read of top_halo_recv (grid[0])
                grid_new[y * Nx + x] = 0.25 * (
                    grid[(y - 1) * Nx + x] + grid[(y + 1) * Nx + x] +
                    grid[y * Nx + (x - 1)] + grid[y * Nx + (x + 1)]
                );
            }
        }

        // Bottom Boundary
        if (rank < size - 1) {
            int y = local_Ny; 
            for (int x = 1; x < Nx - 1; ++x) {
                // LLVM intercepts this read of bot_halo_recv (grid[local_Ny+1])
                grid_new[y * Nx + x] = 0.25 * (
                    grid[(y - 1) * Nx + x] + grid[(y + 1) * Nx + x] +
                    grid[y * Nx + (x - 1)] + grid[y * Nx + (x + 1)]
                );
            }
        }

        // Swap grids
        grid.swap(grid_new);
    }

    double end_time = MDMP_WTIME();

    if (rank == 0) {
        printf("MDMP 2D Heat Equation Benchmark\n");
        printf("Grid Size: %d x %d per rank\n", Nx, local_Ny);
        printf("Iterations: %d\n", iterations);
        printf("Elapsed Time: %f seconds\n", end_time - start_time);
    }

    MDMP_COMM_FINAL();
    return 0;
}
