#include <iostream>
#include <vector>
#include "mdmp_interface.h"

int main() {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    if(size < 2){
        std::cout << "Test not run as we need at least 2 processes to work\n";
        MDMP_COMM_FINAL(); // Safely shut down MPI before early return
        return 0;
    }

    double my_value = (double)(rank + 1); // Ranks 0,1,2,3 -> Values 1,2,3,4
    double total_sum = 0.0;

    MDMP_COMMREGION_BEGIN();

    // Reduce everyone's 'my_value' into Rank 1's 'total_sum' (Root = 1)
    MDMP_REGISTER_REDUCE(&my_value, &total_sum, 1, 1, MDMP_SUM);

    MDMP_COMMIT();

    // PHASE 3: Computation
    // The CPU is completely free to do other things while the tree reduction happens!
    double local_work = my_value * 10.0;

    MDMP_COMMREGION_END();

    if (rank == 1) {
        // Calculate the expected sum: N * (N + 1) / 2
        double expected_sum = (double)(size * (size + 1)) / 2.0;

        if (total_sum == expected_sum) {
            std::cout << "[PASS] Rank 1 successfully registered and reduced total sum: "
                      << total_sum << " == " << expected_sum << std::endl;
        } else {
            std::cout << "[FAIL] Declarative Reduction failed! Expected: "
                      << expected_sum << ", Got: " << total_sum << std::endl;
        }
    }

    MDMP_COMM_FINAL();
    return 0;
}
