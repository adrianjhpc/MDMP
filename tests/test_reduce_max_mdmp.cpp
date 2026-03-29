#include <iostream>
#include <vector>
#include "mdmp_interface.h"

int main() {
    MDMP_COMM_INIT();
    int rank = MDMP_GET_RANK();
    int size = MDMP_GET_SIZE();

    double my_value = (double)(rank + 1); // Ranks 0,1,2,3 -> Values 1,2,3,4
    double max_value = 0.0;

    MDMP_COMMREGION_BEGIN();
    
    // Reduce everyone's 'my_value' into Rank 0's 'max_value'
    MDMP_REDUCE(&my_value, &max_value, 1, 0, MDMP_MAX);
    
    // The CPU is completely free to do other things while the tree reduction happens!
    double local_work = my_value * 10.0; 
    
    MDMP_COMMREGION_END();

    if (rank == 0) {
        double expected_value = (double)size;

        if (max_value == expected_value) {
            std::cout << "[PASS] Root successfully reduced total value: " 
                      << max_value << " == " << expected_value << std::endl;
        } else {
            std::cout << "[FAIL] Reduction failed! Expected: " 
                      << expected_value << ", Got: " << max_value << std::endl;
        }
    }

    MDMP_COMM_FINAL();
    return 0;
}
