#include "mdmp_runtime.h"
#include <stdio.h>
#include <mpi.h>
#include <vector>

// This vector stores all active non-blocking requests for the current region.
static std::vector<MPI_Request> active_requests;
static int global_my_rank = -1; 
static int global_size = 0;

static bool mdmp_debug_mode = false;

extern "C" {
    void __mdmp_marker_init() noexcept {}
    void __mdmp_marker_final() noexcept {}
    void __mdmp_marker_commregion_begin() noexcept {}
    void __mdmp_marker_commregion_end() noexcept {}
    void __mdmp_marker_sync() noexcept {}

    int __mdmp_marker_send(void* buffer, size_t count, int type, size_t byte_size, int sender, int dest, int tag) noexcept { return 0; }
    int __mdmp_marker_recv(void* buffer, size_t count, int type, size_t byte_size, int receiver, int src, int tag) noexcept { return 0; }

    int __mdmp_marker_reduce(void* in_buf, void* out_buf, size_t count, int type, size_t byte_size, int root, int op) noexcept { return 0; }
    int __mdmp_marker_gather(void* send_buf, size_t send_count, void* recv_buf, int type, size_t byte_size, int root) noexcept { return 0; }

    int __mdmp_marker_get_size() noexcept { return 0; }
    int __mdmp_marker_get_rank() noexcept { return 0; }
    double __mdmp_wtime() noexcept { return 0.0; }
}

// Helper to map Enum to MPI Types
MPI_Datatype get_mpi_type(int mdmp_type) {
    switch(mdmp_type) {
        case 0: return MPI_INT;
        case 1: return MPI_DOUBLE;
        case 2: return MPI_FLOAT;
        case 3: return MPI_CHAR;
        default: return MPI_BYTE;
    }
}

MPI_Op get_mpi_op(int op) {
    switch(op) {
        case 0: return MPI_SUM;
        case 1: return MPI_MAX;
        case 2: return MPI_MIN;
        default: return MPI_SUM;
    }
}

// Centralised logging function
void mdmp_log(const char* format, ...) {
    if (!mdmp_debug_mode) return;

    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

void mdmp_set_debug(int enable) noexcept {
    mdmp_debug_mode = (enable != 0);
}

void mdmp_init() {
    const char* env_debug = getenv("MDMP_DEBUG");
    if (env_debug && (env_debug[0] == '1' || env_debug[0] == 't' || env_debug[0] == 'T')) {
        mdmp_debug_mode = true;
    }
    mdmp_log("[MDMP Runtime] Initializing MDMP Environment...\n");
    // Passing NULL is valid in modern MPI and keeps our macro clean!
    MPI_Init(NULL, NULL); 
    MPI_Comm_rank(MPI_COMM_WORLD, &global_my_rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    mdmp_log("[MDMP Rank %d] Initialized (world size %d).\n", global_my_rank, global_size);
}

void mdmp_final() {
    mdmp_log("[MDMP Runtime] Finalizing MDMP Environment...\n");
    MPI_Finalize();
}

void mdmp_commregion_begin() {
    mdmp_log("[MDMP Runtime] Entering communication region.\n");
}

void mdmp_commregion_end() {
    mdmp_log("[MDMP Runtime] Exiting communication region.\n");
}

void mdmp_sync() {
    mdmp_log("[MDMP Runtime] Synchronizing...\n");
    MPI_Barrier(MPI_COMM_WORLD);
}

int mdmp_send(void* buffer, size_t count, int type, int sender_rank, int dest_rank, int tag) {

    // Check for the MDMP_IGNORE flag
    if (sender_rank == -2 || dest_rank == -2) return MPI_REQUEST_NULL;

    // Ignore out of bounds process ranks
    if (sender_rank < 0 || sender_rank >= global_size || 
        dest_rank < 0 || dest_rank >= global_size) {
        return -1;
    }

    // Check if this process should be doing anything
    if (global_my_rank != sender_rank) return MPI_REQUEST_NULL;

    MPI_Request req;
    MPI_Isend(buffer, count, get_mpi_type(type), dest_rank, tag, MPI_COMM_WORLD, &req);
    
    int req_id = active_requests.size();
    active_requests.push_back(req);
    return req_id;
}

int mdmp_recv(void* buffer, size_t count, int type, int receiver_rank, int src_rank, int tag) {
    
    // Check for the MDMP_IGNORE flag
    if (receiver_rank == -2 || src_rank == -2) return MPI_REQUEST_NULL;

    // Ignore out of bounds process ranks
    if (receiver_rank < 0 || receiver_rank >= global_size || 
        src_rank < 0 || src_rank >= global_size) {
        return -1;
    }

    // Check if this process should be doing anything
    if (global_my_rank != receiver_rank) return MPI_REQUEST_NULL;

    MPI_Request req;
    MPI_Irecv(buffer, count, get_mpi_type(type), src_rank, tag, MPI_COMM_WORLD, &req);
    
    int req_id = active_requests.size();
    active_requests.push_back(req);
    return req_id;
}

void mdmp_wait(int req_id) {
    if (req_id < 0 || req_id >= active_requests.size()) return;
    
    mdmp_log("[MDMP Rank %d] Waiting on request %d...\n", global_my_rank, req_id);
    MPI_Wait(&active_requests[req_id], MPI_STATUS_IGNORE);
}

int mdmp_reduce(void* in_buf, void* out_buf, size_t count, int type, int root, int op) {
    MPI_Request req;
    
    // Log the operation if debugging is enabled
    mdmp_log("[MDMP Runtime] Initiating Asynchronous Reduce (Root: %d, Op: %d)...\n", root, op);
    
    // MPI_Ireduce (asynchronous collective)
    MPI_Ireduce(in_buf, out_buf, count, get_mpi_type(type), get_mpi_op(op), root, MPI_COMM_WORLD, &req);
    
    int req_id = active_requests.size();
    active_requests.push_back(req);
    return req_id;
}


int mdmp_gather(void* send_buf, size_t send_count, void* recv_buf, int type, int root) {
    MPI_Request req;
    
    mdmp_log("[MDMP Runtime] Initiating Asynchronous Gather (Root: %d, Count: %zu, Type: %d)...\n", root, send_count, type);
    
    // Explicitly cast send_count to a 32-bit int to satisfy the MPI standard safely
    int mpi_count = (int)send_count;
    
    MPI_Igather(send_buf, mpi_count, get_mpi_type(type), 
                recv_buf, mpi_count, get_mpi_type(type), 
                root, MPI_COMM_WORLD, &req);
    
    int req_id = active_requests.size();
    active_requests.push_back(req);
    return req_id;
}

int mdmp_get_size() {

    return global_size;
}

int mdmp_get_rank() {

    return global_my_rank;

}

double mdmp_wtime() {
    return MPI_Wtime();
}
