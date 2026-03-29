#include "mdmp_runtime.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

static int global_my_rank = -1;
static int global_size = -1;
static int mdmp_debug_enabled = 0;

struct RegisteredMsg {
    void* buffer;
    size_t count;
    int type;
    size_t bytes;
    int rank; 
    int tag;
};

// Tracks custom MPI_Datatypes created during coalescing so we can free them after the wait
static std::vector<MPI_Datatype> custom_types_to_free;

// ==========================================
// Type handling
// ==========================================
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

// ==========================================
// Request management
// ==========================================
static std::vector<MPI_Request> active_requests;
static std::vector<int> free_request_slots;

#define MDMP_GC_TEST_THRESHOLD 1024
#define MDMP_GC_HARD_LIMIT 32768
static int gc_counter = 0;

static void mdmp_garbage_collect_requests() {
    int active_count = 0;
    
    // Pass 1: Non-blocking sweep using MPI_Test
    // This allows the MPI library to its internal handles for completed transfers
    // and ensure we don't run out of resources in that library.
    for (size_t i = 0; i < active_requests.size(); ++i) {
        if (active_requests[i] != MPI_REQUEST_NULL) {
            int flag = 0;
            MPI_Test(&active_requests[i], &flag, MPI_STATUS_IGNORE);
            if (flag) {
                // Transfer is completely finished. Recycle the slot.
                free_request_slots.push_back(i);
            } else {
                active_count++;
            }
        }
    }

    // Pass 2: The Safety Valve
    // If the network is completely saturated and we are hoarding too many 
    // active handles, force a blocking wait to prevent an mpi library crash.
    if (active_count > MDMP_GC_HARD_LIMIT) {
        mdmp_log("[MDMP WARNING] Network saturated! Forcing Hard Flush of %d requests.\n", active_count);
        for (size_t i = 0; i < active_requests.size(); ++i) {
            if (active_requests[i] != MPI_REQUEST_NULL) {
                MPI_Wait(&active_requests[i], MPI_STATUS_IGNORE);
                free_request_slots.push_back(i);
            }
        }
    }
}

static int allocate_request_slot(MPI_Request req) {
    if (req == MPI_REQUEST_NULL) return -2;

    // Trigger background GC periodically
    gc_counter++;
    if (gc_counter >= MDMP_GC_TEST_THRESHOLD) {
        mdmp_garbage_collect_requests();
        gc_counter = 0;
    }

    // Reuse a recycled slot if available
    if (!free_request_slots.empty()) {
        int slot = free_request_slots.back();
        free_request_slots.pop_back();
        active_requests[slot] = req;
        return slot;
    }
    
    // Otherwise, expand the pool
    active_requests.push_back(req);
    return (int)active_requests.size() - 1;
}

static std::vector<RegisteredMsg> send_queue;
static std::vector<RegisteredMsg> recv_queue;

struct RegisteredReduce { void* sendbuf; void* recvbuf; size_t count; int type; size_t bytes; int root; int op; };
struct RegisteredGather { void* sendbuf; size_t sendcount; void* recvbuf; int type; size_t bytes; int root; };

static std::vector<RegisteredReduce> reduce_queue;
static std::vector<RegisteredGather> gather_queue;

struct RegisteredAllreduce { void* sendbuf; void* recvbuf; size_t count; int type; size_t bytes; int op; };
struct RegisteredAllgather { void* sendbuf; size_t count; void* recvbuf; int type; size_t bytes; };


static std::vector<RegisteredAllreduce> allreduce_queue;
static std::vector<RegisteredAllgather> allgather_queue;

void mdmp_init() {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);

   // Read the environment variable once at startup
    const char* env_debug = getenv("MDMP_DEBUG");
    if (env_debug != NULL && atoi(env_debug) > 0) {
        mdmp_debug_enabled = 1;
    }

    mdmp_log("[MDMP Runtime] Initialized Rank %d / %d\n", global_my_rank, global_size);
}

void mdmp_final() {
    MPI_Finalize();
}

int mdmp_get_rank() { return global_my_rank; }
int mdmp_get_size() { return global_size; }
double mdmp_wtime() { return MPI_Wtime(); }
void mdmp_sync()    { MPI_Barrier(MPI_COMM_WORLD); }

void mdmp_set_debug(int enable) {
    mdmp_debug_enabled = enable;
}

void mdmp_commregion_begin() {}
void mdmp_commregion_end() {}

void mdmp_abort(int error_code) {
    // Flush any pending logs so we know exactly who called this function
    fflush(stdout); 
    mdmp_log("[MDMP FATAL] Rank %d called ABORT with error code %d. Terminating...\n", global_my_rank, error_code);
    
    // MDMP uses MPI under the hood, so we still trigger the hardware abort
    MPI_Abort(MPI_COMM_WORLD, error_code);
}

void mdmp_wait(int req_id) {
    mdmp_log("[MDMP] Rank %d WAITING on request ID: %d\n", global_my_rank, req_id);
    if (req_id == -1) {
        // Bulk Wait for Declarative API
        if (!active_requests.empty()) {
            MPI_Waitall((int)active_requests.size(), active_requests.data(), MPI_STATUSES_IGNORE);
            active_requests.clear();
            free_request_slots.clear();
            
            // Safe to free custom coalesced types now that the transfers are complete
            for (auto T : custom_types_to_free) {
                if (T != MPI_DATATYPE_NULL) MPI_Type_free(&T);
            }
            custom_types_to_free.clear();
        }
    } else if (req_id >= 0 && req_id < (int)active_requests.size()) {
        // Individual Wait for Imperative API
        if (active_requests[req_id] != MPI_REQUEST_NULL) {
            MPI_Wait(&active_requests[req_id], MPI_STATUS_IGNORE);
            active_requests[req_id] = MPI_REQUEST_NULL;
            free_request_slots.push_back(req_id);
        }
    }
    mdmp_log("[MDMP] Rank %d WAIT COMPLETE for request ID: %d\n", global_my_rank, req_id);
}

int mdmp_send(void* buf, size_t c, int t, size_t bytes, int s, int d, int tag) {
    if (d < 0 || d >= global_size || global_my_rank != s) return -2;
    MPI_Request req;
    
    // If it's a custom struct (MPI_BYTE), the true count is the total bytes
    int actual_count = (t == 4) ? (int)bytes : (int)c;
    
    MPI_Isend(buf, actual_count, get_mpi_type(t), d, tag, MPI_COMM_WORLD, &req);
    return allocate_request_slot(req);
}

int mdmp_recv(void* buf, size_t c, int t, size_t bytes, int r, int s, int tag) {
    if (s < 0 || s >= global_size || global_my_rank != r) return -2;
    MPI_Request req;
    
    int actual_count = (t == 4) ? (int)bytes : (int)c;
    
    MPI_Irecv(buf, actual_count, get_mpi_type(t), s, tag, MPI_COMM_WORLD, &req);
    return allocate_request_slot(req);
}

// Collectives
int mdmp_reduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes, int root, int op) {
    MPI_Request req;
    const void* final_sendbuf = sendbuf;
    if (sendbuf == recvbuf && global_my_rank == root) final_sendbuf = MPI_IN_PLACE;

    int actual_count = (type == 4) ? (int)bytes : (int)count;
    MPI_Ireduce(final_sendbuf, recvbuf, actual_count, get_mpi_type(type), get_mpi_op(op), root, MPI_COMM_WORLD, &req);
    return allocate_request_slot(req);
}

int mdmp_gather(void* sb, size_t sc, void* rb, int t, size_t bytes, int root) {
    MPI_Request req;
    const void* final_sb = sb;
    if (sb == rb && global_my_rank == root) final_sb = MPI_IN_PLACE;

    int actual_count = (t == 4) ? (int)bytes : (int)sc;
    MPI_Igather(final_sb, actual_count, get_mpi_type(t), rb, actual_count, get_mpi_type(t), root, MPI_COMM_WORLD, &req);
    return allocate_request_slot(req);
}

int mdmp_allreduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes, int op) {
    MPI_Request req;
    const void* final_sendbuf = sendbuf;
    if (sendbuf == recvbuf) final_sendbuf = MPI_IN_PLACE;

    int actual_count = (type == 4) ? (int)bytes : (int)count;
    MPI_Op mpi_op = (MPI_Op)(intptr_t)op; 
    MPI_Iallreduce(final_sendbuf, recvbuf, actual_count, get_mpi_type(type), mpi_op, MPI_COMM_WORLD, &req);
    return allocate_request_slot(req);
}

int mdmp_allgather(void* sb, size_t c, void* rb, int t, size_t bytes) {
    MPI_Request req;
    const void* final_sb = sb;
    if (sb == rb) final_sb = MPI_IN_PLACE;

    int actual_count = (t == 4) ? (int)bytes : (int)c;
    MPI_Iallgather(final_sb, actual_count, get_mpi_type(t), rb, actual_count, get_mpi_type(t), MPI_COMM_WORLD, &req);
    return allocate_request_slot(req);
}

void mdmp_register_send(void* b, size_t c, int t, size_t bytes, int s, int d, int tag) {
    if (d >= 0 && d < global_size && global_my_rank == s) {
        send_queue.push_back({b, c, t, bytes, d, tag}); // Store 'bytes'
    }
}

void mdmp_register_recv(void* b, size_t c, int t, size_t bytes, int r, int s, int tag) {
    if (s >= 0 && s < global_size && global_my_rank == r) {
        recv_queue.push_back({b, c, t, bytes, s, tag}); // Store 'bytes'
    }
}

void mdmp_register_reduce(void* sb, void* rb, size_t c, int t, size_t bytes, int root, int op) {
    reduce_queue.push_back({sb, rb, c, t, bytes, root, op});
    mdmp_log("[MDMP] Rank %d queue REDUCE. Queue size %zu\n", global_my_rank, reduce_queue.size());
}

void mdmp_register_gather(void* sb, size_t sc, void* rb, int t, size_t bytes, int root) {
    gather_queue.push_back({sb, sc, rb, t, bytes, root});
    mdmp_log("[MDMP] Rank %d queue GATHER. Queue size %zu\n", global_my_rank, gather_queue.size());
}

void mdmp_register_allreduce(void* sb, void* rb, size_t c, int t, size_t bytes, int op) {
    allreduce_queue.push_back({sb, rb, c, t, bytes, op});
    mdmp_log("[MDMP] Rank %d queue ALLREDUCE. Queue size %zu\n", global_my_rank, allreduce_queue.size());
}

void mdmp_register_allgather(void* sb, size_t c, void* rb, int t, size_t bytes) {
    allgather_queue.push_back({sb, c, rb, t, bytes});
    mdmp_log("[MDMP] Rank %d queue ALLGATHER. Queue size %zu\n", global_my_rank, allgather_queue.size());
}

int mdmp_commit() {
    mdmp_log("[MDMP] Rank %d ENTERING COMMIT. Processing %zu Sends, %zu Recvs, %zu Allreduces, %zu Allgathers\n", 
             global_my_rank, send_queue.size(), recv_queue.size(), allreduce_queue.size(), allgather_queue.size());
    // Process P2P queues
    auto ProcessQueue = [](std::vector<RegisteredMsg>& queue, bool isSend) {
        if (queue.empty()) return;

        // Sort by both rank and tag to group identical destinations/tags together
        std::stable_sort(queue.begin(), queue.end(), [](const RegisteredMsg& a, const RegisteredMsg& b) {
            if (a.rank != b.rank) return a.rank < b.rank;
            return a.tag < b.tag;
        });

        size_t i = 0;
        while (i < queue.size()) {
            int peer = queue[i].rank;
            int current_tag = queue[i].tag; // Grab the tag for this group
            size_t j = i + 1;
            
            // Group messages that share both the same peer and same tag
            while (j < queue.size() && queue[j].rank == peer && queue[j].tag == current_tag) {
                j++;
            }

            size_t count = j - i;
            
            // Fire individually if there is only 1 message
            if (count == 1) { 
                MPI_Request req;
                int actual_count = (queue[i].type == 4) ? (int)queue[i].bytes : (int)queue[i].count;
                
                if (isSend) MPI_Isend(queue[i].buffer, actual_count, get_mpi_type(queue[i].type), peer, current_tag, MPI_COMM_WORLD, &req);
                else        MPI_Irecv(queue[i].buffer, actual_count, get_mpi_type(queue[i].type), peer, current_tag, MPI_COMM_WORLD, &req);
                allocate_request_slot(req);
            } 
            // 4. Fire the Zero-Copy Hardware Datatype if there are multiple
            else {
                std::vector<int> blens(count);
                std::vector<MPI_Aint> disps(count);
                for (size_t k = 0; k < count; k++) {
                    blens[k] = (int)queue[i + k].bytes;
                    MPI_Get_address(queue[i + k].buffer, &disps[k]);
                }
                MPI_Datatype ntype;
                
                MPI_Type_create_hindexed((int)count, blens.data(), disps.data(), MPI_BYTE, &ntype);
                MPI_Type_commit(&ntype);
                custom_types_to_free.push_back(ntype); 

                MPI_Request req;
                if (isSend) MPI_Isend(MPI_BOTTOM, 1, ntype, peer, current_tag, MPI_COMM_WORLD, &req);
                else        MPI_Irecv(MPI_BOTTOM, 1, ntype, peer, current_tag, MPI_COMM_WORLD, &req);
                allocate_request_slot(req);
            }
            i = j;
        }
    };

    ProcessQueue(recv_queue, false);
    ProcessQueue(send_queue, true);

    for (auto& r : reduce_queue) mdmp_reduce(r.sendbuf, r.recvbuf, r.count, r.type, r.bytes, r.root, r.op);
    for (auto& g : gather_queue) mdmp_gather(g.sendbuf, g.sendcount, g.recvbuf, g.type, g.bytes, g.root);
    
    for (auto& ar : allreduce_queue) mdmp_allreduce(ar.sendbuf, ar.recvbuf, ar.count, ar.type, ar.bytes, ar.op);
    for (auto& ag : allgather_queue) mdmp_allgather(ag.sendbuf, ag.count, ag.recvbuf, ag.type, ag.bytes);
 
    send_queue.clear();
    recv_queue.clear();

    reduce_queue.clear();
    gather_queue.clear();

    allreduce_queue.clear();
    allgather_queue.clear();

    mdmp_log("[MDMP] Rank %d EXITED COMMIT. Hardware datatypes dispatched.\n", global_my_rank);

    return -1;
}
