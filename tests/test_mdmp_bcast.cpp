#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mdmp_interface.h"

int main(int argc, char** argv) {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    const int N = 1000;
    int errors = 0;

    int root1 = 0;
    std::vector<int> int_data(N, -1); // Initialize everyone to -1

    if (rank == root1) {
        for (int i = 0; i < N; ++i) int_data[i] = i * 2;
    }

    MDMP_COMMREGION_BEGIN();
    MDMP_BCAST(int_data.data(), N, root1);
    MDMP_COMMREGION_END();

    for (int i = 0; i < N; ++i) {
        if (int_data[i] != i * 2) errors++;
    }

    if (rank != root1) {
        if (errors == 0) printf("[PASS] Rank %d successfully received INT broadcast from Root %d\n", rank, root1);
        else             printf("[FAIL] Rank %d found %d errors in INT broadcast\n", rank, errors);
    }

    if (size > 1) {
        int root2 = 1;
        std::vector<double> double_data(N, -1.0);

        if (rank == root2) {
            for (int i = 0; i < N; ++i) double_data[i] = i * 3.14;
        }

        MDMP_COMMREGION_BEGIN();
        MDMP_BCAST(double_data.data(), N, root2);
        MDMP_COMMREGION_END();

        for (int i = 0; i < N; ++i) {
            // Using a small epsilon for floating point comparison just to be safe
            if (double_data[i] < (i * 3.14) - 0.001 || double_data[i] > (i * 3.14) + 0.001) {
                errors++;
            }
        }

        if (rank != root2) {
            if (errors == 0) printf("[PASS] Rank %d successfully received DOUBLE broadcast from Root %d\n", rank, root2);
            else             printf("[FAIL] Rank %d found %d errors in DOUBLE broadcast\n", rank, errors);
        }
    }

    MDMP_COMM_FINAL();
    
    // CTest relies on the exit code. 0 = Pass, >0 = Fail.
    return errors > 0 ? 1 : 0;
}
