#include <iostream>
#include <vector>
#include "mdmp_interface.h"

int main() {
    MDMP_COMM_INIT();

    const int size = 1000;
    std::vector<double> data(size, 1.0);

    MDMP_COMMREGION_BEGIN();

    MDMP_REGISTER_SEND(data.data(), size, ((MDMP_RANK % 2 == 0 && MDMP_RANK < 3) ? MDMP_RANK : MDMP_IGNORE), 1, 0);

    MDMP_REGISTER_RECV(data.data(), size, 1, 0, 0);
    MDMP_REGISTER_RECV(data.data(), size, 1, 2, 0);

    MDMP_COMMIT();

    for (int i = 0; i < size; ++i) {
        data[i] = data[i] * 2.0;
    }

    MDMP_COMMREGION_END();

    MDMP_COMM_FINAL();
    return 0;
}
