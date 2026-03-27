#include <iostream>
#include <vector>
#include <iomanip>

#include "mdmp_pragma_interface.h"
#include "mdmp_runtime.h"

int main() {
    int mdmp_rank, mdmp_size;

    MDMP_COMM_INIT();
    mdmp_rank = MDMP_GET_RANK();
    mdmp_size = MDMP_GET_SIZE();

    if (mdmp_rank == 0) {
        std::cout << "=== MDMP Declarative Coalescing Test ===" << std::endl;
        std::cout << "Running on " << mdmp_size << " ranks." << std::endl;
    }

    const int size = 10000;
    std::vector<double> pressure(size, 0.0);
    std::vector<double> temperature(size, 0.0);

    // Rank 0 initializes the data to send
    if (mdmp_rank == 0) {
        for (int i = 0; i < size; ++i) {
            pressure[i] = static_cast<double>(i) * 1.5;
            temperature[i] = static_cast<double>(i) * 2.5;
        }
    }

    MDMP_COMMREGION_BEGIN();
    
    // Rank 0 registers two sends to Rank 1
    MDMP_REGISTER_SEND(pressure.data(), size, 0, 1, 100); 
    MDMP_REGISTER_SEND(temperature.data(), size, 0, 1, 101);

    // Rank 1 registers two receives from Rank 0
    MDMP_REGISTER_RECV(pressure.data(), size, 1, 0, 100);
    MDMP_REGISTER_RECV(temperature.data(), size, 1, 0, 101);    

    MDMP_COMMIT();

    double local_math_sum = 0.0;
    for (int i = 0; i < size; ++i) {
        local_math_sum += (i * 0.01); 
    }

    MDMP_COMMREGION_END();

    MDMP_COMM_SYNC();

    // Verify the zero-copy batch transfer worked
    if (mdmp_rank == 1) {
        std::cout << "[Rank 1] Math completed. Local sum: " << local_math_sum << std::endl;
        std::cout << "[Rank 1] Verification - Pressure[10]: " << std::fixed << std::setprecision(2) << pressure[10] << " (Expected: 15.00)" << std::endl;
        std::cout << "[Rank 1] Verification - Temp[10]: " << std::fixed << std::setprecision(2) << temperature[10] << " (Expected: 25.00)" << std::endl;
        
        if (pressure[10] == 15.00 && temperature[10] == 25.00) {
            std::cout << "\nSUCCESS: Inspector-Executor flawlessly coalesced and delivered the data!" << std::endl;
        } else {
            std::cout << "\nFAILED: Data mismatch." << std::endl;
        }
    }

    MDMP_COMM_FINAL();
    return 0;
}
