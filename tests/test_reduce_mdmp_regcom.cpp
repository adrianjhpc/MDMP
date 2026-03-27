#include <iostream>
#include <vector>
#include "mdmp_pragma_interface.h"

int main() {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    double my_value = (double)(rank + 1); // Ranks 0,1,2,3 -> Values 1,2,3,4
    double total_sum = 0.0;

    MDMP_COMMREGION_BEGIN();

    // Register the collective reduction. All ranks must participate
    MDMP_REGISTER_REDUCE(&my_value, &total_sum, 1, 0, MDMP_SUM);

    // The runtime dispatches the non-blocking MPI_Ireduce operation.
    MDMP_COMMIT();

    // The CPU is completely free to do other things while the tree reduction happens.
    // The LLVM pass should sink the Waitall *after* this local work.
    double local_work = my_value * 10.0;

    MDMP_COMMREGION_END();
    // ========================================================================

    if (rank == 0) {
        // Calculate the expected sum: N * (N + 1) / 2
        double expected_sum = (double)(size * (size + 1)) / 2.0;

        if (total_sum == expected_sum) {
            std::cout << "[PASS] Root successfully registered and reduced total sum: "
                      << total_sum << " == " << expected_sum << std::endl;
        } else {
            std::cout << "[FAIL] Declarative Reduction failed! Expected: "
                      << expected_sum << ", Got: " << total_sum << std::endl;
        }
    }

    MDMP_COMM_FINAL();
    return 0;
}
