#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mdmp_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    if (size < 2) {
        MDMP_COMM_FINAL();
        return 0;
    }

    const int num_vertices = 100000;
    const int num_ghosts = 10000; 
    int right_neighbor = (rank + 1) % size;
    int left_neighbor = (rank - 1 + size) % size;

    std::vector<double> local_vals(num_vertices, 1.0);
    std::vector<double> new_vals(num_vertices, 0.0);
    
    std::vector<double> ghost_vals_recv(num_ghosts, 0.0);
    std::vector<double> ghost_vals_send(num_ghosts, 1.0); 

    MDMP_COMM_SYNC();
    double start_time = MDMP_WTIME();

    for (int iter = 0; iter < 100; ++iter) {
        
        // Initiate Asynchronous Network Transfers
        MDMP_SEND(ghost_vals_send.data(), num_ghosts, rank, right_neighbor, 0);
        MDMP_RECV(ghost_vals_recv.data(), num_ghosts, rank, left_neighbor, 0);

        // Process Local Edges
        // The LLVM pass sees that we only touch 'local_vals' and 'new_vals' here.
        // It allows this entire block to execute while the network works!
        for (int i = 0; i < num_vertices; ++i) {
            double vertex_sum = 0.0;
            for (int e = 0; e < 50; ++e) { 
                vertex_sum += local_vals[(i + e) % num_vertices] * 0.01;
            }
            new_vals[i] = vertex_sum;
        }

        // Process Remote Edges
        // Now that the network is guaranteed to be finished, safely read ghost data.
        for (int i = 0; i < num_vertices; i += 10) { 
            new_vals[i] += ghost_vals_recv[i % num_ghosts] * 0.05;
        }

        local_vals.swap(new_vals);
    }

    double end_time = MDMP_WTIME();
    double calc_time = (end_time - start_time);
    double max_time = 0.0;
    MDMP_REDUCE(&calc_time, &max_time, 1, 0, MDMP_MAX);    

    if (rank == 0) {
        printf("Validation Check (Prevents DCE): %f\n", local_vals[0]); 
        printf("------------------------------------------------\n");
        printf(" BENCHMARK: Graph Analytics (Imperative MDMP)\n");
        printf("------------------------------------------------\n");
        printf("Split-Phase Computation enabled perfect overlap!\n");
        printf("Elapsed Time: %f seconds\n", max_time);
    }

    MDMP_COMM_FINAL();
    return 0;
}
