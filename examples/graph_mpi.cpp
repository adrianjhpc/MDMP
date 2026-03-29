#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        MPI_Finalize();
        return 0;
    }

    const int num_vertices = 10000;
    const int num_ghosts = 1000;
    int right_neighbor = (rank + 1) % size;
    int left_neighbor = (rank - 1 + size) % size;

    std::vector<double> local_vals(num_vertices, 1.0);
    std::vector<double> new_vals(num_vertices, 0.0);
    std::vector<double> ghost_vals_recv(num_ghosts, 0.0);
    std::vector<double> ghost_vals_send(num_ghosts, 1.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int iter = 0; iter < 100; ++iter) {
        
        MPI_Request reqs[2];
        
        MPI_Isend(ghost_vals_send.data(), num_ghosts, MPI_DOUBLE, right_neighbor, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(ghost_vals_recv.data(), num_ghosts, MPI_DOUBLE, left_neighbor, 0, MPI_COMM_WORLD, &reqs[1]);

        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        for (int i = 0; i < num_vertices; ++i) {
            double vertex_sum = 0.0;

            // Compute local edges
            for (int e = 0; e < 50; ++e) { 
                vertex_sum += local_vals[(i + e) % num_vertices] * 0.01;
            }

            // Compute remote edges
            if (i % 10 == 0) { 
                vertex_sum += ghost_vals_recv[i % num_ghosts] * 0.05;
            }

            new_vals[i] = vertex_sum;
        }

        local_vals.swap(new_vals);
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Validation Check (Prevents DCE): %f\n", local_vals[0]); 
    }

    if (rank == 0) {
        printf("------------------------------------------------\n");
        printf(" BENCHMARK: Graph Analytics (Raw MPI)\n");
        printf("------------------------------------------------\n");
        printf("Elapsed Time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
