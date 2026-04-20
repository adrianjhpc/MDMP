#include <mpi.h>

#include <iostream>

#include "spmv_common.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    BenchmarkConfig cfg = parse_config(argc, argv);
    SpMVProblem P = build_problem(rank, size, cfg);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int iter = 0; iter < cfg.iters; ++iter) {
        zero_vector(P.local_y);

        MPI_Request reqs[4];
        int req_count = 0;

        // Post receives first
        if (P.have_left) {
            MPI_Irecv(P.recv_left.data(),
                      P.left_count,
                      MPI_DOUBLE,
                      P.left_rank,
                      SpMVProblem::TAG_SEND_LAST,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }

        if (P.have_right) {
            MPI_Irecv(P.recv_right.data(),
                      P.right_count,
                      MPI_DOUBLE,
                      P.right_rank,
                      SpMVProblem::TAG_SEND_FIRST,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }

        // Pack halos, then send
        pack_halos(P);

        if (P.have_left) {
            MPI_Isend(P.send_left.data(),
                      P.left_count,
                      MPI_DOUBLE,
                      P.left_rank,
                      SpMVProblem::TAG_SEND_FIRST,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }

        if (P.have_right) {
            MPI_Isend(P.send_right.data(),
                      P.right_count,
                      MPI_DOUBLE,
                      P.right_rank,
                      SpMVProblem::TAG_SEND_LAST,
                      MPI_COMM_WORLD,
                      &reqs[req_count++]);
        }

        // Overlappable local-only work
        compute_spmv(P.local_A, P.local_x, P.local_y);

        if (req_count > 0) {
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        }

        unpack_halos(P);

        // Ghost-dependent work
        compute_spmv(P.ghost_A, P.ghost_x, P.local_y);
    }

    double local_time = MPI_Wtime() - t0;
    double max_time = 0.0;

    double local_chk = checksum(P.local_y);
    double global_chk = 0.0;

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_chk, &global_chk, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "SpMV MPI benchmark on " << size << " ranks\n";
        std::cout << "  local_rows = " << cfg.local_rows
                  << ", halo_width = " << std::min(cfg.halo_width, std::max(1, cfg.local_rows / 8))
                  << ", iters = " << cfg.iters << "\n";
        std::cout << "  elapsed (max rank) = " << max_time << " s\n";
        std::cout << "  checksum           = " << global_chk << "\n";
    }

    MPI_Finalize();
    return 0;
}

