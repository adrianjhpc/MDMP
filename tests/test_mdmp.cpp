// tests/test_mdmp.cpp
#include <iostream>
#include <vector>
#include <iomanip>

// Include the interface (for the macros) and the runtime (for linking)
#include "mdmp_pragma_interface.h"
#include "mdmp_runtime.h"

extern "C" {
    void __mdmp_marker_init() {}
    void __mdmp_marker_final() {}
    void __mdmp_marker_commregion_begin() {}
    void __mdmp_marker_commregion_end() {}
    void __mdmp_marker_sync() {}
    int __mdmp_marker_send(void* buffer, size_t count, int type, size_t byte_size, int sender, int dest) { return 0; }
    int __mdmp_marker_recv(void* buffer, size_t count, int type, size_t byte_size, int receiver, int src) { return 0; }
    int __mdmp_marker_get_rank() { return 0; }
    int __mdmp_marker_get_size() { return 0;}
}

int main() {

    int mdmp_rank, mdmp_size;

    MDMP_COMM_INIT();

    mdmp_rank = MDMP_GET_RANK();
    mdmp_size = MDMP_GET_SIZE();

    std::cout << "=== Simple MDMP Test ===" << std::endl;
    std::cout << "Rank " << mdmp_rank << " of " << mdmp_size << std::endl;

    const int size = 10000;
    std::vector<double> data(size);
    std::vector<double> result(size);

    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<double>(i);
    }

    MDMP_COMMREGION_BEGIN();
    MDMP_SEND(data.data(), size, 0, 1); // Rank 0 sends to Rank 1
    MDMP_RECV(result.data(), size, 1, 0); // Rank 1 recvs from Rank 0
    MDMP_SEND(result.data(), size, 1, 0); // Rank 1 sends to Rank 0
    MDMP_RECV(result.data(), size, 0, 1); // Rank 0 recvs from Rank 1

    for (int i = 0; i < size; ++i) {
        result[i] = result[i] * result[i] * result[i];
    }
    MDMP_COMMREGION_END();

    MDMP_COMM_SYNC();

    MDMP_COMM_FINAL();

    std::cout << "First result: " << std::fixed << std::setprecision(6) << result[10] << std::endl;
    std::cout << "Simple MDMP test completed successfully!" << std::endl;
    
    return 0;
}
