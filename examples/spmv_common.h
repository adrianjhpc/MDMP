#ifndef SPMV_BENCH_COMMON_H
#define SPMV_BENCH_COMMON_H

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <vector>

struct CSRMatrix {
    std::vector<double> vals;
    std::vector<int> cols;
    std::vector<int> row_ptrs;
};

struct BenchmarkConfig {
    int local_rows = 100000;
    int halo_width = 2048;
    int iters = 500;
};

inline BenchmarkConfig parse_config(int argc, char** argv) {
    BenchmarkConfig cfg;

    if (argc > 1) cfg.local_rows = std::max(1, std::atoi(argv[1]));
    if (argc > 2) cfg.halo_width = std::max(0, std::atoi(argv[2]));
    if (argc > 3) cfg.iters      = std::max(1, std::atoi(argv[3]));

    return cfg;
}

struct SpMVProblem {
    // Tag meanings:
    // TAG_SEND_FIRST: sender's first halo_width values, sent to its left neighbour
    // TAG_SEND_LAST : sender's last  halo_width values, sent to its right neighbour
    static constexpr int TAG_SEND_FIRST = 100;
    static constexpr int TAG_SEND_LAST  = 101;

    int rank = 0;
    int size = 1;

    bool have_left = false;
    bool have_right = false;

    int left_rank = -1;
    int right_rank = -1;

    int local_rows = 0;
    int left_count = 0;
    int right_count = 0;

    CSRMatrix local_A; // local-only contribution: uses local_x
    CSRMatrix ghost_A; // ghost-only contribution: uses ghost_x

    std::vector<double> local_x;
    std::vector<double> local_y;
    std::vector<double> ghost_x;

    std::vector<double> send_left;
    std::vector<double> send_right;
    std::vector<double> recv_left;
    std::vector<double> recv_right;
};

inline void zero_vector(std::vector<double>& v) {
    std::fill(v.begin(), v.end(), 0.0);
}

inline void compute_spmv(const CSRMatrix& A,
                         const std::vector<double>& x,
                         std::vector<double>& y) {
    const int rows = static_cast<int>(A.row_ptrs.size()) - 1;
    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (int p = A.row_ptrs[i]; p < A.row_ptrs[i + 1]; ++p) {
            sum += A.vals[p] * x[A.cols[p]];
        }
        y[i] += sum;
    }
}

inline void pack_halos(SpMVProblem& P) {
    if (P.have_left && P.left_count > 0) {
        std::copy_n(P.local_x.begin(), P.left_count, P.send_left.begin());
    }

    if (P.have_right && P.right_count > 0) {
        std::copy_n(P.local_x.end() - P.right_count,
                    P.right_count,
                    P.send_right.begin());
    }
}

inline void unpack_halos(SpMVProblem& P) {
    if (P.have_left && P.left_count > 0) {
        std::copy(P.recv_left.begin(), P.recv_left.end(), P.ghost_x.begin());
    }

    if (P.have_right && P.right_count > 0) {
        std::copy(P.recv_right.begin(),
                  P.recv_right.end(),
                  P.ghost_x.begin() + P.left_count);
    }
}

inline double checksum(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0);
}

inline SpMVProblem build_problem(int rank, int size, const BenchmarkConfig& cfg) {
    SpMVProblem P;
    P.rank = rank;
    P.size = size;
    P.local_rows = cfg.local_rows;

    P.have_left  = (size > 1);
    P.have_right = (size > 1);

    P.left_rank  = P.have_left  ? (rank - 1 + size) % size : -1;
    P.right_rank = P.have_right ? (rank + 1) % size        : -1;

    const int halo = (size > 1)
        ? std::min(cfg.halo_width, std::max(1, cfg.local_rows / 8))
        : 0;

    P.left_count  = P.have_left  ? halo : 0;
    P.right_count = P.have_right ? halo : 0;

    P.local_x.resize(P.local_rows);
    P.local_y.assign(P.local_rows, 0.0);
    P.ghost_x.assign(P.left_count + P.right_count, 0.0);

    P.send_left.assign(P.left_count, 0.0);
    P.send_right.assign(P.right_count, 0.0);
    P.recv_left.assign(P.left_count, 0.0);
    P.recv_right.assign(P.right_count, 0.0);

    // Slightly varying initial values to avoid a too-trivial state.
    for (int i = 0; i < P.local_rows; ++i) {
        P.local_x[i] = 1.0 + 0.0001 * ((rank * P.local_rows + i) % 97);
    }

    // Reserve some space for a moderately compute-heavy local matrix.
    P.local_A.vals.reserve(static_cast<size_t>(P.local_rows) * 7);
    P.local_A.cols.reserve(static_cast<size_t>(P.local_rows) * 7);
    P.ghost_A.vals.reserve(static_cast<size_t>(P.left_count + P.right_count));
    P.ghost_A.cols.reserve(static_cast<size_t>(P.left_count + P.right_count));

    P.local_A.row_ptrs.reserve(P.local_rows + 1);
    P.ghost_A.row_ptrs.reserve(P.local_rows + 1);

    P.local_A.row_ptrs.push_back(0);
    P.ghost_A.row_ptrs.push_back(0);

    auto add_local = [&](int col, double val) {
        P.local_A.cols.push_back(col);
        P.local_A.vals.push_back(val);
    };

    auto add_ghost = [&](int col, double val) {
        P.ghost_A.cols.push_back(col);
        P.ghost_A.vals.push_back(val);
    };

    // Synthetic split:
    // - local_A: a 7-point 1D banded stencil restricted to local_x
    // - ghost_A: one ghost contribution for each left-boundary and right-boundary row
    for (int i = 0; i < P.local_rows; ++i) {
        add_local(i, 6.0);
        if (i - 1 >= 0)             add_local(i - 1, -1.0);
        if (i + 1 < P.local_rows)   add_local(i + 1, -1.0);
        if (i - 2 >= 0)             add_local(i - 2, -0.5);
        if (i + 2 < P.local_rows)   add_local(i + 2, -0.5);
        if (i - 4 >= 0)             add_local(i - 4, -0.25);
        if (i + 4 < P.local_rows)   add_local(i + 4, -0.25);

        if (P.have_left && i < P.left_count) {
            // This row depends on left neighbour data.
            // Left ghost buffer occupies ghost_x[0 .. left_count)
            add_ghost(i, -0.75);
        }

        if (P.have_right && i >= P.local_rows - P.right_count) {
            // Right ghost buffer occupies ghost_x[left_count .. left_count + right_count)
            int right_ghost_col = P.left_count + (i - (P.local_rows - P.right_count));
            add_ghost(right_ghost_col, -0.75);
        }

        P.local_A.row_ptrs.push_back(static_cast<int>(P.local_A.vals.size()));
        P.ghost_A.row_ptrs.push_back(static_cast<int>(P.ghost_A.vals.size()));
    }

    return P;
}

#endif

