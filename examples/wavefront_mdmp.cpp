#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include "mdmp_interface.h"

// A heavy math function to simulate real scientific workloads and simulate network delay
double heavy_math(double input) {
    double result = input;
    for(int i = 0; i < 1000; i++) { result = sin(result + 0.1) * cos(result - 0.1); }
    return result;
}

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    if (size < 2) return 0;

    const int num_elements = 20000;
    std::vector<double> local_data(num_elements, rank + 1.0);
    double remote_ghost = 0.0;

    MDMP_COMM_SYNC();
    double start_time = MDMP_WTIME();

    if (rank == 0) {
        // Rank 0 has work to do before sending 
        for (int i = 0; i < num_elements; ++i) { local_data[i] = heavy_math(local_data[i]); }
        MDMP_SEND(&local_data[num_elements-1], 1, rank, 1, 0);
    } 
    else if (rank == 1) {
        MDMP_RECV(&remote_ghost, 1, rank, 0, 0);

        for (int i = 0; i < num_elements-1; ++i) {
            
            // Most of the work is purely local.
            local_data[i] = heavy_math(local_data[i]); 
        }    
        // Only the very last element needs the remote ghost cell.
        local_data[num_elements-1] += remote_ghost; 
       
    }

    double end_time = MDMP_WTIME();

    if (rank == 1) {
        printf("------------------------------------------------\n");
        printf(" BENCHMARK: Imbalanced Wavefront (Imperative)\n");
        printf("------------------------------------------------\n");
        printf("Elapsed Time: %f seconds\n", end_time - start_time);
        printf("Validation (Prevents DCE): %f\n", local_data[0]);
    }

    MDMP_COMM_FINAL();
    return 0;
}
