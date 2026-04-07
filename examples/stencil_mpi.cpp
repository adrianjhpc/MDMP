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

        MPI_Sendrecv(&grid[0], N, MPI_DOUBLE, up_neighbor, 0,
                     halo_bottom_recv.data(), N, MPI_DOUBLE, down_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&grid[(N - 1) * N], N, MPI_DOUBLE, down_neighbor, 1,
                     halo_top_recv.data(), N, MPI_DOUBLE, up_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int y = 0; y < N; ++y) {
            for (int x = 1; x < N - 1; ++x) {
                int i = y * N + x;
                double up_val = (y == 0) ? halo_top_recv[x] : grid[i - N];
                double down_val = (y == N - 1) ? halo_bottom_recv[x] : grid[i + N];
                next_grid[i] = 0.25 * (grid[i - 1] + grid[i + 1] + up_val + down_val);
            }
        }

        grid.swap(next_grid);
    }

    double end_time = MPI_Wtime();

    double calc_time = (end_time - start_time);
    double max_time = 0.0;
    MPI_Reduce(&calc_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 1, MPI_COMM_WORLD);
    
    if (rank == 1) {
        printf("------------------------------------------------\n");
        printf(" BENCHMARK: 2D Heat Equation (Raw MPI)\n");
        printf("------------------------------------------------\n");
        printf("Grid Size per Rank: %d x %d\n", N, N);
        printf("Elapsed Time: %f seconds\n", max_time);
        printf("Validation: %f\n", grid[N/2 + N/2]); 
    }

    MPI_Finalize();
    return 0;
}
