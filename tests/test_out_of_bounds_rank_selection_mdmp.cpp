#include <iostream>
#include <vector>
#include "mdmp_interface.h"
#include "mdmp_runtime.h"

int main() {
    MDMP_COMM_INIT();

    const int size = 10;
    // Every rank starts with its own ID in the array
    std::vector<double> data(size, static_cast<double>(MDMP_RANK));

    MDMP_COMMREGION_BEGIN();

    // SHIFT RIGHT PATTERN
    // Every rank sends to the right (Rank + 1)
    // The library will automatically ignore the send for the very last rank!
    MDMP_SEND(data.data(), size, MDMP_RANK, MDMP_RANK + 1, 0);

    // Every rank receives from the left (Rank - 1)
    // The library will automatically ignore the receive for Rank 0!
    MDMP_RECV(data.data(), size, MDMP_RANK, MDMP_RANK - 1, 0);

    for (int i = 0; i < size; ++i) {
        // Just some dummy compute
        double temp = data[i] * 2.0;
    }

    MDMP_COMMREGION_END();
    
    // Print the result to prove it worked
    std::cout << "[Rank " << MDMP_RANK << "] First element is now: " << data[0] << std::endl;

    MDMP_COMM_FINAL();
    return 0;
}
