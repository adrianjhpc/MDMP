#include "mdmp_runtime.h"
#include <stdio.h>
#include <mpi.h>
#include <vector>

// This vector stores all active non-blocking requests for the current region.
static std::vector<MPI_Request> active_requests;
static int global_my_rank = -1; 
static int global_size = 0;

// Helper to map Enum to MPI Types
MPI_Datatype get_mpi_type(int mdmp_type) {
    switch(mdmp_type) {
        case 0: return MPI_INT;
        case 1: return MPI_FLOAT;
        case 2: return MPI_DOUBLE;
        case 3: return MPI_CHAR;
        default: return MPI_BYTE;
    }
}

void mdmp_init() {
    printf("[MDMP Runtime] Initializing MDMP Environment...\n");
    // Passing NULL is valid in modern MPI and keeps our macro clean!
    MPI_Init(NULL, NULL); 
    MPI_Comm_rank(MPI_COMM_WORLD, &global_my_rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    printf("[MDMP Rank %d] Initialized (world size %d).\n", global_my_rank, global_size);
}

void mdmp_final() {
    printf("[MDMP Runtime] Finalizing MDMP Environment...\n");
    MPI_Finalize();
}

void mdmp_commregion_begin() {
    printf("[MDMP Runtime] Entering communication region.\n");
}

void mdmp_commregion_end() {
    printf("[MDMP Runtime] Exiting communication region.\n");
}

void mdmp_sync() {
    printf("[MDMP Runtime] Synchronizing...\n");
    MPI_Barrier(MPI_COMM_WORLD);
}

int mdmp_send(void* buffer, size_t count, int type, int sender_rank, int dest_rank) {
    if (global_my_rank != sender_rank) return -1;

    MPI_Request req;
    MPI_Isend(buffer, count, get_mpi_type(type), dest_rank, 0, MPI_COMM_WORLD, &req);
    
    int req_id = active_requests.size();
    active_requests.push_back(req);
    return req_id;
}

int mdmp_recv(void* buffer, size_t count, int type, int receiver_rank, int src_rank) {
    if (global_my_rank != receiver_rank) return -1;

    MPI_Request req;
    MPI_Irecv(buffer, count, get_mpi_type(type), src_rank, 0, MPI_COMM_WORLD, &req);
    
    int req_id = active_requests.size();
    active_requests.push_back(req);
    return req_id;
}

void mdmp_wait(int req_id) {
    if (req_id < 0 || req_id >= active_requests.size()) return;
    
    printf("[MDMP Rank %d] Waiting on request %d...\n", global_my_rank, req_id);
    MPI_Wait(&active_requests[req_id], MPI_STATUS_IGNORE);
}

int mdmp_get_size() {

    return global_size;
}

int mdmp_get_rank() {

    return global_my_rank;

}
