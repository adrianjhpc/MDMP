// example_usage.cpp
#include <mpi.h>
#include "mdmp_runtime.h"

void compute_something(){

    return;

}

void process_data(){

     return;

}

void example_function() {
    // Example of using MDMP pragmas
#pragma mdmp_commbegin

    // Some computation that can overlap with communication
    compute_something();
    
#pragma mpmd_commend

    // More computation
    process_data();
}

int main(int argc, char **argv){


     example_function();

     return 0;

}

