#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        
        if (rank > 0) {
            MPI_Sendrecv(top_halo_send, Nx, MPI_DOUBLE, rank - 1, 0,
                         top_halo_recv, Nx, MPI_DOUBLE, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(bot_halo_send, Nx, MPI_DOUBLE, rank + 1, 1,
                         bot_halo_recv, Nx, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int y = 2; y <= local_Ny - 1; ++y) {
            for (int x = 1; x < Nx - 1; ++x) {
                grid_new[y * Nx + x] = 0.25 * (
                    grid[(y - 1) * Nx + x] + grid[(y + 1) * Nx + x] +
                    grid[y * Nx + (x - 1)] + grid[y * Nx + (x + 1)]
                );
            }
        }

        // Top Boundary
        if (rank > 0) {
            int y = 1; 
            for (int x = 1; x < Nx - 1; ++x) {
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
                grid_new[y * Nx + x] = 0.25 * (
                    grid[(y - 1) * Nx + x] + grid[(y + 1) * Nx + x] +
                    grid[y * Nx + (x - 1)] + grid[y * Nx + (x + 1)]
                );
            }
        }

        // Swap grids
        grid.swap(grid_new);
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Standard MPI 2D Heat Equation Benchmark\n");
        printf("Grid Size: %d x %d per rank\n", Nx, local_Ny);
        printf("Iterations: %d\n", iterations);
        printf("Elapsed Time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
