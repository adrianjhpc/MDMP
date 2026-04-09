#include "mdmp_interface.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>

// Minimal Compressed Sparse Row (CSR) Matrix
struct CSRMatrix {
    std::vector<double> vals;
    std::vector<int> cols;
    std::vector<int> row_ptrs;
};

// Standard SpMV kernel: y = A * x
void compute_spmv(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y) {
    for (size_t i = 0; i < A.row_ptrs.size() - 1; ++i) {
        double sum = 0.0;
        for (int j = A.row_ptrs[i]; j < A.row_ptrs[i+1]; ++j) {
            sum += A.vals[j] * x[A.cols[j]];
        }
        y[i] += sum;
    }
}

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    // =====================================================================
    // 1. MOCK SETUP: Unstructured Graph & Data Allocation
    // =====================================================================
    const int local_rows = 100000;
    const int max_iter = 10000;

    CSRMatrix local_A, ghost_A;
    std::vector<double> local_x(local_rows, 1.0);
    std::vector<double> local_y(local_rows, 0.0);

    // Create a symmetric 1D ring topology (connect to left and right)
    std::vector<int> neighbours;
    if (size > 1) {
        neighbours.push_back((rank + 1) % size);             // Right neighbour
        neighbours.push_back((rank - 1 + size) % size);      // Left neighbour
        // If size == 2, left and right are the same rank, so remove the duplicate
        if (size == 2) neighbours.pop_back();
    }

    int num_actual_neighbours = neighbours.size();
    std::vector<int> send_counts;
    std::vector<int> recv_counts;
    std::vector<std::vector<int>> send_indices;

    for (int i = 0; i < num_actual_neighbours; ++i) {
        int count = 1500; // Static size for the mock
        send_counts.push_back(count);
        recv_counts.push_back(count);

        std::vector<int> indices(count);
        for(int j=0; j<count; ++j) indices[j] = rand() % local_rows;
        send_indices.push_back(indices);
    }

    // Allocate communication buffers
    std::vector<std::vector<double>> send_buffers(num_actual_neighbours);
    std::vector<std::vector<double>> recv_buffers(num_actual_neighbours);
    for (int i = 0; i < num_actual_neighbours; ++i) {
        send_buffers[i].resize(send_counts[i]);
        recv_buffers[i].resize(recv_counts[i]);
    }

    // Ghost vector size is sum of all recv_counts
    int total_ghosts = 0;
    std::vector<std::vector<int>> recv_indices(num_actual_neighbours);
    for (int i = 0; i < num_actual_neighbours; ++i) {
        recv_indices[i].resize(recv_counts[i]);
        for(int j=0; j<recv_counts[i]; ++j) {
            recv_indices[i][j] = total_ghosts++;
        }
    }
    std::vector<double> ghost_x(total_ghosts, 0.0);

    // Setup mock matrices to prevent segfaults in compute
    local_A.row_ptrs.resize(local_rows + 1, 0);
    ghost_A.row_ptrs.resize(local_rows + 1, 0);

    MDMP_COMM_SYNC();

    // =====================================================================
    // 2. MDMP INSPECTOR PHASE (Outside the hot loop)
    // =====================================================================
    MDMP_COMMREGION_BEGIN();
    for (int i = 0; i < num_actual_neighbours; ++i) {
        MDMP_REGISTER_SEND(send_buffers[i].data(), send_counts[i], rank, neighbours[i], 0);
        MDMP_REGISTER_RECV(recv_buffers[i].data(), recv_counts[i], rank, neighbours[i], 0);
    }
    MDMP_COMMREGION_END();

    double start_time = MDMP_WTIME();

    // =====================================================================
    // 3. THE HOT SOLVER LOOP (DECLARATIVE EXECUTOR)
    // =====================================================================
    for (int iter = 0; iter < max_iter; ++iter) {
        
        // Pack Sends (Pure Memory, No opaque C calls)
        for (int i = 0; i < num_actual_neighbours; ++i) {
            for (int j = 0; j < send_counts[i]; ++j) {
                send_buffers[i][j] = local_x[send_indices[i][j]];
            }
        }

        // Batched Dispatch
        MDMP_COMMIT(); 

        // Overlapped Compute
        compute_spmv(local_A, local_x, local_y);

        // Unpack & Finish Compute
        for (int i = 0; i < num_actual_neighbours; ++i) {
            for (int j = 0; j < recv_counts[i]; ++j) {
                ghost_x[recv_indices[i][j]] = recv_buffers[i][j];
            }
        }
        compute_spmv(ghost_A, ghost_x, local_y);
    }

    double end_time = MDMP_WTIME();

    double calc_time = (end_time - start_time);
    double max_time = 0.0;
    MDMP_REDUCE(&calc_time, &max_time, 1, 0, MDMP_MAX);

    if (rank == 0) {
        std::cout << "SpMV example run on " << size << " processes\n";
        std::cout << "MDMP Declarative SpMV completed in " << max_time << " seconds.\n";
    }

    MDMP_COMM_FINAL();
    return 0;
}
