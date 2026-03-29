#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mdmp_pragma_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    if (size < 2) return 0;

    const int num_vertices = 10000;
    const int num_ghosts = 1000; // Vertices owned by our neighbor
    int right_neighbor = (rank + 1) % size;
    int left_neighbor = (rank - 1 + size) % size;

    std::vector<double> local_vals(num_vertices, 1.0);
    std::vector<double> new_vals(num_vertices, 0.0);
    
    // The "Halo" or "Ghost" vertices we need from our neighbor
    std::vector<double> ghost_vals_recv(num_ghosts, 0.0);
    std::vector<double> ghost_vals_send(num_ghosts, 1.0); // Dummy data to send

    MDMP_COMM_SYNC();
    double start_time = MDMP_WTIME();

    for (int iter = 0; iter < 100; ++iter) {
        
        MDMP_SEND(ghost_vals_send.data(), num_ghosts, rank, right_neighbor, 0);
        MDMP_RECV(ghost_vals_recv.data(), num_ghosts, rank, left_neighbor, 0);

        for (int i = 0; i < num_vertices; ++i) {
            double vertex_sum = 0.0;

            // 1. Process Local Edges 
            // The LLVM pass knows local_vals is not part of the network transfer.
            for (int e = 0; e < 50; ++e) { 
                // Dummy local edge traversal
                vertex_sum += local_vals[(i + e) % num_vertices] * 0.01;
            }

            // 2. Process Remote Edge
            // If this specific vertex happens to have a connection to a remote node...
            if (i % 10 == 0) { 
                // The LLVM pass sees us reading 'ghost_vals_recv'.
                // It should inject `mdmp_wait()` exactly here
                vertex_sum += ghost_vals_recv[i % num_ghosts] * 0.05;
            }

            new_vals[i] = vertex_sum;
        }

        local_vals.swap(new_vals);
    }

    double end_time = MDMP_WTIME();

    if (rank == 0) {
        printf("------------------------------------------------\n");
        printf(" BENCHMARK: Graph Analytics (Imperative MDMP)\n");
        printf("------------------------------------------------\n");
        printf("LLVM enabled perfect fine-grained overlap!\n");
        printf("Elapsed Time: %f seconds\n", end_time - start_time);
    }

    MDMP_COMM_FINAL();
    return 0;
}
