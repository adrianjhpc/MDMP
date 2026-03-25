#include <iostream>
#include <vector>
#include "mdmp_pragma_interface.h"

int main() {
    MDMP_COMM_INIT();

    const int size = 1000;
    std::vector<double> data(size, 1.0);

    MDMP_COMMREGION_BEGIN();

    // Ranks 0 and 2 will send to Rank 1. 
    // Everyone else (like Rank 3, 4, etc.) evaluates to MDMP_IGNORE and safely skips.
    MDMP_SEND(data.data(), size, ((MDMP_RANK % 2 == 0 && MDMP_RANK < 3) ? MDMP_RANK : MDMP_IGNORE), 1, 0);

    // Rank 1 receives from Rank 0, and then receives from Rank 2
    MDMP_RECV(data.data(), size, 1, 0, 0);
    MDMP_RECV(data.data(), size, 1, 2, 0);

    for (int i = 0; i < size; ++i) {
        data[i] = data[i] * 2.0; 
    }

    MDMP_COMMREGION_END();
    MDMP_COMM_FINAL();
    return 0;
}
