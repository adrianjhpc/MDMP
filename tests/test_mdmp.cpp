// test_mdmp_simple.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

// Include MDMP runtime header
#include "mdmp_runtime.h"

// Simple test to verify MDMP pragmas work
int main() {
    std::cout << "=== Simple MDMP Test ===" << std::endl;
    
    const int size = 10000;
    std::vector<double> data(size);
    std::vector<double> result(size);
    
    // Initialize data
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<double>(i) / size;
    }
    
    // Apply MDMP pragma
    // TODO
    for (int i = 0; i < size; ++i) {
        result[i] = data[i] * data[i] * data[i];
    }
    
    // Verify first few results
    std::cout << "First 5 results:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "result[" << i << "] = " << std::fixed << std::setprecision(6) 
                  << result[i] << std::endl;
    }
    
    // Verify last few results
    std::cout << "Last 5 results:" << std::endl;
    for (int i = size - 5; i < size; ++i) {
        std::cout << "result[" << i << "] = " << std::fixed << std::setprecision(6) 
                  << result[i] << std::endl;
    }
    
    std::cout << "Simple MDMP test completed successfully!" << std::endl;
    return 0;
}

