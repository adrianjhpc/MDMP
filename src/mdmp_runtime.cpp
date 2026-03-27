#include "mdmp_runtime.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

static int global_my_rank = -1;
static int global_size = -1;

struct RegisteredMsg {
    void* buffer;
    size_t count;
    int type;
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

static int allocate_request_slot(MPI_Request req) {
    if (req == MPI_REQUEST_NULL) return -2;
    if (!free_request_slots.empty()) {
        int slot = free_request_slots.back();
        free_request_slots.pop_back();
        active_requests[slot] = req;
        return slot;
    }
    active_requests.push_back(req);
    return (int)active_requests.size() - 1;
}

static std::vector<RegisteredMsg> send_queue;
static std::vector<RegisteredMsg> recv_queue;

struct RegisteredReduce { void* sendbuf; void* recvbuf; size_t count; int type; int root; int op; };
struct RegisteredGather { void* sendbuf; size_t sendcount; void* recvbuf; int type; int root; };

static std::vector<RegisteredReduce> reduce_queue;
static std::vector<RegisteredGather> gather_queue;

void mdmp_init() {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_size);
}

void mdmp_final() {
    MPI_Finalize();
}

int mdmp_get_rank() { return global_my_rank; }
int mdmp_get_size() { return global_size; }
double mdmp_wtime() { return MPI_Wtime(); }
void mdmp_sync()    { MPI_Barrier(MPI_COMM_WORLD); }
void mdmp_commregion_begin() {}
void mdmp_commregion_end() {}

void mdmp_wait(int req_id) {
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
}

int mdmp_send(void* buf, size_t c, int t, int s, int d, int tag) {
    if (d < 0 || d >= global_size || global_my_rank != s) return -2;
    MPI_Request req;
    MPI_Isend(buf, (int)c, get_mpi_type(t), d, tag, MPI_COMM_WORLD, &req);
    return allocate_request_slot(req);
}

int mdmp_recv(void* buf, size_t c, int t, int r, int s, int tag) {
    if (s < 0 || s >= global_size || global_my_rank != r) return -2;
    MPI_Request req;
    MPI_Irecv(buf, (int)c, get_mpi_type(t), s, tag, MPI_COMM_WORLD, &req);
    return allocate_request_slot(req);
}

// (Collectives mdmp_reduce and mdmp_gather remain standard as per your previous code)
int mdmp_reduce(void* sb, void* rb, size_t c, int t, int root, int op) {
    MPI_Request req;
    MPI_Ireduce(sb, rb, (int)c, get_mpi_type(t), get_mpi_op(op), root, MPI_COMM_WORLD, &req);
    return allocate_request_slot(req);
}

int mdmp_gather(void* sb, size_t sc, void* rb, int t, int root) {
    MPI_Request req;
    MPI_Igather(sb, (int)sc, get_mpi_type(t), rb, (int)sc, get_mpi_type(t), root, MPI_COMM_WORLD, &req);
    return allocate_request_slot(req);
}

void mdmp_register_send(void* b, size_t c, int t, int s, int d, int tag) {
    if (d >= 0 && d < global_size && global_my_rank == s) send_queue.push_back({b, c, t, d, tag});
}

void mdmp_register_recv(void* b, size_t c, int t, int r, int s, int tag) {
    if (s >= 0 && s < global_size && global_my_rank == r) recv_queue.push_back({b, c, t, s, tag});
}

void mdmp_register_reduce(void* sb, void* rb, size_t c, int t, int root, int op) {
    reduce_queue.push_back({sb, rb, c, t, root, op});
}

void mdmp_register_gather(void* sb, size_t sc, void* rb, int t, int root) {
    gather_queue.push_back({sb, sc, rb, t, root});
}

int mdmp_commit() {
    // Process P2P queues
    auto ProcessQueue = [](std::vector<RegisteredMsg>& queue, bool isSend) {
        if (queue.empty()) return;

        // Sort by rank to group messages for the same neighbor
        std::sort(queue.begin(), queue.end(), [](const RegisteredMsg& a, const RegisteredMsg& b) {
            return a.rank < b.rank;
        });

        size_t i = 0;
        while (i < queue.size()) {
            int peer = queue[i].rank;
            size_t j = i + 1;
            
            // Group messages for the same peer
            while (j < queue.size() && queue[j].rank == peer) {
                j++;
            }

            size_t count = j - i;
            // Only coalesce if multiple messages exist for the SAME peer AND SAME tag
            // If tags differ, we must send them individually but still in this loop
            if (count == 1 || queue[i].tag != queue[i+1].tag) { 
                for(size_t k = i; k < j; ++k) {
                    MPI_Request req;
                    if (isSend) MPI_Isend(queue[k].buffer, (int)queue[k].count, get_mpi_type(queue[k].type), peer, queue[k].tag, MPI_COMM_WORLD, &req);
                    else        MPI_Irecv(queue[k].buffer, (int)queue[k].count, get_mpi_type(queue[k].type), peer, queue[k].tag, MPI_COMM_WORLD, &req);
                    allocate_request_slot(req);
                }
            } else {
                // Coalesce Path (Multiple messages, same peer, same tag)
                std::vector<int> blens(count);
                std::vector<MPI_Aint> disps(count);
                for (size_t k = 0; k < count; k++) {
                    blens[k] = (int)queue[i + k].count;
                    MPI_Get_address(queue[i + k].buffer, &disps[k]);
                }
                MPI_Datatype ntype;
                MPI_Type_create_hindexed((int)count, blens.data(), disps.data(), get_mpi_type(queue[i].type), &ntype);
                MPI_Type_commit(&ntype);
                custom_types_to_free.push_back(ntype);

                MPI_Request req;
                if (isSend) MPI_Isend(MPI_BOTTOM, 1, ntype, peer, queue[i].tag, MPI_COMM_WORLD, &req);
                else        MPI_Irecv(MPI_BOTTOM, 1, ntype, peer, queue[i].tag, MPI_COMM_WORLD, &req);
                allocate_request_slot(req);
            }
            i = j;
        }
    };

    ProcessQueue(recv_queue, false);
    ProcessQueue(send_queue, true);

    for (auto& r : reduce_queue) mdmp_reduce(r.sendbuf, r.recvbuf, r.count, r.type, r.root, r.op);
    for (auto& g : gather_queue) mdmp_gather(g.sendbuf, g.sendcount, g.recvbuf, g.type, g.root);

    send_queue.clear();
    recv_queue.clear();
    reduce_queue.clear();
    gather_queue.clear();

    return -1;
}
