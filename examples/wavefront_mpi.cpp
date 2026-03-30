#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <mpi.h>

double heavy_math(double input) {
    double result = input;
    for(int i = 0; i < 1000; i++) { result = sin(result + 0.1) * cos(result - 0.1); }
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) { MPI_Finalize(); return 0; }

    const int num_elements = 20000;
    std::vector<double> local_data(num_elements, rank + 1.0);
    double remote_ghost = 0.0;
    MPI_Request req;

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    if (rank == 0) {
        for (int i = 0; i < num_elements; ++i) { local_data[i] = heavy_math(local_data[i]); }
        MPI_Isend(&local_data[num_elements-1], 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } 
    else if (rank == 1) {
        MPI_Irecv(&remote_ghost, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req);
        
        // The user waits here because they need the ghost cell eventually.
        // Rank 1 sits totally idle while Rank 0 finishes working
        MPI_Wait(&req, MPI_STATUS_IGNORE);

        for (int i = 0; i < num_elements; ++i) {
            local_data[i] = heavy_math(local_data[i]); 
            if (i == num_elements - 1) { local_data[i] += remote_ghost; }
        }
    }

    double end_time = MPI_Wtime();

    if (rank == 1) {
        printf("------------------------------------------------\n");
        printf(" BENCHMARK: Imbalanced Wavefront (Raw MPI)\n");
        printf("------------------------------------------------\n");
        printf("Elapsed Time: %f seconds\n", end_time - start_time);
        printf("Validation (Prevents DCE): %f\n", local_data[0]);
    }

    MPI_Finalize();
    return 0;
}
