#include "mdmp_runtime.h"
#include <mpi.h>
#include <vector>
#include <map>
#include <cstdio>
#include <cstdlib>

// ==========================================
// GLOBAL STATE & UTILITIES
// ==========================================
static int global_my_rank = -1;
static int global_size = -1;

// Standard MPI Type Mapper (Extend based on your specific framework needs)
static inline MPI_Datatype get_mpi_type(int type_code) {
    switch(type_code) {
        case 0: return MPI_INT;
        case 1: return MPI_DOUBLE;
        case 2: return MPI_FLOAT;
        case 3: return MPI_CHAR;
        case 4: return MPI_BYTE;
        default: return MPI_BYTE;
    }
}

static inline MPI_Op get_mpi_op(int op_code) {
    switch(op_code) {
        case 0: return MPI_SUM;
        case 1: return MPI_MAX;
        case 2: return MPI_MIN;
        case 3: return MPI_PROD;
        default: return MPI_SUM;
    }
}

#define DEBUG_LOG 0
#define mdmp_log(...) do { if (DEBUG_LOG) { printf(__VA_ARGS__); fflush(stdout); } } while(0)

// ==========================================
// Manage async requests
// ==========================================
static std::vector<MPI_Request> active_requests;
static std::vector<int> free_request_slots;

static int allocate_request_slot(MPI_Request req) {
    if (!free_request_slots.empty()) {
        int slot = free_request_slots.back();
        free_request_slots.pop_back();
        active_requests[slot] = req;
        return slot;
    }
    active_requests.push_back(req);
    return active_requests.size() - 1;
}

// ============================================
// Queues for the regsiter/commit functionality
// ============================================
struct RegisteredMsg {
    void* buffer;
    size_t count;
    int type;
    int rank; // Peer rank (Dest for Send, Src for Recv)
    int tag;
};

static std::vector<RegisteredMsg> send_queue;
static std::vector<RegisteredMsg> recv_queue;

struct RegisteredReduce { 
    void* sendbuf; 
    void* recvbuf; 
    size_t count; 
    int type; 
    int root; 
    int op; 
};
struct RegisteredGather { 
    void* sendbuf; 
    size_t sendcount; 
    void* recvbuf; 
    int type; 
    int root; 
};

static std::vector<RegisteredReduce> reduce_queue;
static std::vector<RegisteredGather> gather_queue;

// ==========================================
// Helper functionality
// ==========================================
void mdmp_init() {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    mdmp_log("[MDMP Runtime] Initialized Rank %d / %d\n", global_my_rank, global_size);
}

void mdmp_final() {
    MPI_Finalize();
}

int mdmp_get_rank() { return global_my_rank; }
int mdmp_get_size() { return global_size; }
double mdmp_wtime() { return MPI_Wtime(); }
void mdmp_sync()    { MPI_Barrier(MPI_COMM_WORLD); }

void mdmp_commregion_begin() {
    // Empty hook for the LLVM Pass to target
}

void mdmp_commregion_end() {
    // Empty hook for the LLVM Pass to target
}

// ==========================================
// Wait engine
// ==========================================
void mdmp_wait(int req_id)  {
    if (req_id == -1) {
        // If users have manually created sends and receives then we do a bulk wait here 
        if (!active_requests.empty()) {
            mdmp_log("[MDMP Runtime] CFG Engine triggered bulk wait on %zu requests...\n", active_requests.size());
            MPI_Waitall(active_requests.size(), active_requests.data(), MPI_STATUSES_IGNORE);
            active_requests.clear();
            free_request_slots.clear();
        }
    } else if (req_id >= 0 && req_id < (int)active_requests.size()) {
        // If this is the register/commit approach, just wait on the single request that has been passed
        if (active_requests[req_id] != MPI_REQUEST_NULL) {
            MPI_Wait(&active_requests[req_id], MPI_STATUS_IGNORE);
            active_requests[req_id] = MPI_REQUEST_NULL; 
            free_request_slots.push_back(req_id);
        }
    }
}

// ==========================================
// Imperative functionality
// ==========================================
int mdmp_send(void* buffer, size_t count, int type, int sender_rank, int dest_rank, int tag) {
    if (dest_rank == -2 || dest_rank < 0 || dest_rank >= global_size) return MPI_REQUEST_NULL;
    if (global_my_rank != sender_rank) return MPI_REQUEST_NULL;

    MPI_Request req;
    MPI_Isend(buffer, count, get_mpi_type(type), dest_rank, tag, MPI_COMM_WORLD, &req);
    return allocate_request_slot(req);
}

int mdmp_recv(void* buffer, size_t count, int type, int receiver_rank, int src_rank, int tag) {
    if (src_rank == -2 || src_rank < 0 || src_rank >= global_size) return MPI_REQUEST_NULL;
    if (global_my_rank != receiver_rank) return MPI_REQUEST_NULL;

    MPI_Request req;
    MPI_Irecv(buffer, count, get_mpi_type(type), src_rank, tag, MPI_COMM_WORLD, &req);
    return allocate_request_slot(req);
}

int mdmp_reduce(void* sendbuf, void* recvbuf, size_t count, int type, int root_rank, int op) {
    MPI_Request req;
    MPI_Ireduce(sendbuf, recvbuf, count, get_mpi_type(type), get_mpi_op(op), root_rank, MPI_COMM_WORLD, &req);
    return allocate_request_slot(req);
}

int mdmp_gather(void* sendbuf, size_t sendcount, void* recvbuf, int type, int root_rank) {
    MPI_Request req;
    // Note: recvcount in MPI_Gather is the number of elements *per* receiving rank
    MPI_Igather(sendbuf, sendcount, get_mpi_type(type), recvbuf, sendcount, get_mpi_type(type), root_rank, MPI_COMM_WORLD, &req);
    return allocate_request_slot(req);
}

// ==========================================
// Declarative functionality
// ==========================================
void mdmp_register_send(void* buffer, size_t count, int type, int sender_rank, int dest_rank, int tag) {
    if (dest_rank == -2 || dest_rank < 0 || dest_rank >= global_size) return;
    if (global_my_rank != sender_rank) return; 
    
    send_queue.push_back({buffer, count, type, dest_rank, tag});
}

void mdmp_register_recv(void* buffer, size_t count, int type, int receiver_rank, int src_rank, int tag) {
    if (src_rank == -2 || src_rank < 0 || src_rank >= global_size) return;
    if (global_my_rank != receiver_rank) return; 
    
    recv_queue.push_back({buffer, count, type, src_rank, tag});
}

void mdmp_register_reduce(void* sendbuf, void* recvbuf, size_t count, int type, int root_rank, int op) {
    reduce_queue.push_back({sendbuf, recvbuf, count, type, root_rank, op});
}

void mdmp_register_gather(void* sendbuf, size_t sendcount, void* recvbuf, int type, int root_rank) {
    gather_queue.push_back({sendbuf, sendcount, recvbuf, type, root_rank});
}


int mdmp_commit()  {
    mdmp_log("[MDMP Runtime] Committing Communication Region...\n");

    auto ProcessQueue = [](std::vector<RegisteredMsg>& queue, bool isSend) {
        std::map<int, std::vector<RegisteredMsg>> buckets;
        for (auto& msg : queue) {
            buckets[msg.rank].push_back(msg);
        }

        for (auto& pair : buckets) {
            int peer = pair.first;
            auto& msgs = pair.second;

            if (msgs.size() == 1) {
                // Standard Dispatch
                MPI_Request req;
                if (isSend) {
                    MPI_Isend(msgs[0].buffer, msgs[0].count, get_mpi_type(msgs[0].type), peer, msgs[0].tag, MPI_COMM_WORLD, &req);
                } else {
                    MPI_Irecv(msgs[0].buffer, msgs[0].count, get_mpi_type(msgs[0].type), peer, msgs[0].tag, MPI_COMM_WORLD, &req);
                }
                allocate_request_slot(req);
            } 
            else {
                // Zero-Copy Hardware Coalescing
                mdmp_log("[MDMP Runtime] Coalescing %zu messages for Rank %d\n", msgs.size(), peer);
                
                std::vector<int> block_lengths(msgs.size());
                std::vector<MPI_Aint> displacements(msgs.size());
                
                for (size_t i = 0; i < msgs.size(); i++) {
                    block_lengths[i] = (int)msgs[i].count;
                    MPI_Get_address(msgs[i].buffer, &displacements[i]);
                }

                MPI_Datatype merged_type;
                MPI_Type_create_hindexed(msgs.size(), block_lengths.data(), displacements.data(), get_mpi_type(msgs[0].type), &merged_type);
                MPI_Type_commit(&merged_type);

                MPI_Request req;
                // Dispatch using absolute memory offsets mapped to MPI_BOTTOM
                if (isSend) {
                    MPI_Isend(MPI_BOTTOM, 1, merged_type, peer, msgs[0].tag, MPI_COMM_WORLD, &req);
                } else {
                    MPI_Irecv(MPI_BOTTOM, 1, merged_type, peer, msgs[0].tag, MPI_COMM_WORLD, &req);
                }
                
                MPI_Type_free(&merged_type);
                allocate_request_slot(req);
            }
        }
    };

    // Process the point to point communications (receives first then sends)
    ProcessQueue(recv_queue, false);
    ProcessQueue(send_queue, true);

    // Process the collective queues
    for (auto& r : reduce_queue) {
        MPI_Request req;
        MPI_Ireduce(r.sendbuf, r.recvbuf, r.count, get_mpi_type(r.type), get_mpi_op(r.op), r.root, MPI_COMM_WORLD, &req);
        allocate_request_slot(req);
    }
    for (auto& g : gather_queue) {
        MPI_Request req;
        MPI_Igather(g.sendbuf, g.sendcount, get_mpi_type(g.type), g.recvbuf, g.sendcount, get_mpi_type(g.type), g.root, MPI_COMM_WORLD, &req);
        allocate_request_slot(req);
    }
    
    // Clear the wait queues
    reduce_queue.clear();
    gather_queue.clear();

    send_queue.clear();
    recv_queue.clear();

    // Return Magic Batch ID for the LLVM pass to Waitall
    return -1;
}
