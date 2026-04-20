#include <mpi.h>

#include <iostream>

#include "mdmp_interface.h"
#include "spmv_common.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();

    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    BenchmarkConfig cfg = parse_config(argc, argv);
    SpMVProblem P = build_problem(rank, size, cfg);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int iter = 0; iter < cfg.iters; ++iter) {
        zero_vector(P.local_y);

        MDMP_COMMREGION_BEGIN();

        // Explicit fixed callsites: better fit for the imperative pass
        if (P.have_left) {
            MDMP_RECV(P.recv_left.data(),
                      P.left_count,
                      rank,
                      P.left_rank,
                      SpMVProblem::TAG_SEND_LAST);
        }

        if (P.have_right) {
            MDMP_RECV(P.recv_right.data(),
                      P.right_count,
                      rank,
                      P.right_rank,
                      SpMVProblem::TAG_SEND_FIRST);
        }

        pack_halos(P);

        if (P.have_left) {
            MDMP_SEND(P.send_left.data(),
                      P.left_count,
                      rank,
                      P.left_rank,
                      SpMVProblem::TAG_SEND_FIRST);
        }

        if (P.have_right) {
            MDMP_SEND(P.send_right.data(),
                      P.right_count,
                      rank,
                      P.right_rank,
                      SpMVProblem::TAG_SEND_LAST);
        }

        // Local-only work to overlap with comm
        compute_spmv(P.local_A, P.local_x, P.local_y);

        MDMP_COMMREGION_END();

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
        std::cout << "SpMV MDMP Imperative benchmark on " << size << " ranks\n";
        std::cout << "  local_rows = " << cfg.local_rows
                  << ", halo_width = " << std::min(cfg.halo_width, std::max(1, cfg.local_rows / 8))
                  << ", iters = " << cfg.iters << "\n";
        std::cout << "  elapsed (max rank) = " << max_time << " s\n";
        std::cout << "  checksum           = " << global_chk << "\n";
    }

    MDMP_COMM_FINAL();
    return 0;
}

