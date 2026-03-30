#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include "mdmp_interface.h"

// Heavy math function to simulate real scientific workloads
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
        // Rank 0 has work to do before it sends
        for (int i = 0; i < num_elements; ++i) { local_data[i] = heavy_math(local_data[i]); }
        
        MDMP_COMMREGION_BEGIN();
        MDMP_REGISTER_SEND(&local_data[num_elements-1], 1, rank, 1, 0);
        MDMP_COMMIT();
        MDMP_COMMREGION_END();
    } 
    else if (rank == 1) {
        MDMP_COMMREGION_BEGIN();
        MDMP_REGISTER_RECV(&remote_ghost, 1, rank, 0, 0);
        MDMP_COMMIT();
        
        // The declarative fence forces a bulk wait here
        MDMP_COMMREGION_END();

        // By the time we reach this loop, the overlap opportunity is completely gone.
        for (int i = 0; i < num_elements-1; ++i) {
            local_data[i] = heavy_math(local_data[i]); 
         }
         local_data[num_elements-1] += remote_ghost; 
         
    }

    double end_time = MDMP_WTIME();

    if (rank == 1) {
        printf("------------------------------------------------\n");
        printf(" BENCHMARK: Imbalanced Wavefront (Declarative)\n");
        printf("------------------------------------------------\n");
        printf("Elapsed Time: %f seconds\n", end_time - start_time);
        printf("Validation (Prevents DCE): %f\n", local_data[0]);
    }

    MDMP_COMM_FINAL();
    return 0;
}
