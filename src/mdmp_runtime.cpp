#include "mdmp_runtime.h"

static int global_my_rank = -1;
static int global_size = -1;
static int mdmp_debug_enabled = 0;

MPI_Comm mdmp_comm; 
std::atomic<bool> mdmp_runtime_active{false};
std::thread mdmp_progress_thread;
std::mutex mdmp_mpi_mutex;
bool mdmp_owns_mpi = false;

MPI_Request mdmp_request_pool[MDMP_MAX_REQUESTS];
uint32_t mdmp_req_counter = 0;

MPI_Request mdmp_declarative_requests[MDMP_MAX_DECLARATIVE_REQS];
int mdmp_declarative_req_count = 0;

MPI_Datatype mdmp_declarative_types_to_free[MDMP_MAX_DECLARATIVE_REQS];
int mdmp_declarative_type_count = 0;  

// --- NEW: Unbounded Declarative ID Tracking ---
uint32_t mdmp_decl_req_counter = 0;
std::unordered_map<int, int> decl_to_mpi_req;
// ----------------------------------------------

struct RegisteredMsg { void* buffer; size_t count; int type; size_t bytes; int rank; int tag; int req_id; };
static std::vector<RegisteredMsg> send_queue;
static std::vector<RegisteredMsg> recv_queue;

struct RegisteredReduce { void* sendbuf; void* recvbuf; size_t count; int type; size_t bytes; int root; int op; int req_id; };
static std::vector<RegisteredReduce> reduce_queue;

struct RegisteredGather { void* sendbuf; size_t sendcount; void* recvbuf; int type; size_t bytes; int root; int req_id; };
static std::vector<RegisteredGather> gather_queue;

struct RegisteredAllreduce { void* sendbuf; void* recvbuf; size_t count; int type; size_t bytes; int op; int req_id; };
static std::vector<RegisteredAllreduce> allreduce_queue;

struct RegisteredAllgather { void* sendbuf; size_t count; void* recvbuf; int type; size_t bytes; int req_id; };
static std::vector<RegisteredAllgather> allgather_queue;

struct RegisteredBcast { void* buffer; size_t count; int type; size_t bytes; int root; int req_id; };
static std::vector<RegisteredBcast> bcast_queue;


// ==========================================
// Type handling & Background Engine
// ==========================================
static inline MPI_Datatype get_mpi_type(int type_code) {
  switch(type_code) {
  case 0: return MPI_INT; case 1: return MPI_DOUBLE; case 2: return MPI_FLOAT;
  case 3: return MPI_CHAR; case 4: return MPI_BYTE; default: return MPI_BYTE;
  }
}

static inline MPI_Op get_mpi_op(int op_code) {
  if (op_code > 10) return (MPI_Op)(intptr_t)op_code;
  switch(op_code) {
  case 0: return MPI_SUM; case 1: return MPI_MAX; case 2: return MPI_MIN;
  case 3: return MPI_PROD; default: return MPI_SUM;
  }
}

void mdmp_progress_loop() {
  int flag;
  while (mdmp_runtime_active) {
    {
      std::lock_guard<std::mutex> lock(mdmp_mpi_mutex);
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, mdmp_comm, &flag, MPI_STATUS_IGNORE);
    }
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}

void mdmp_progress() {
  int flag;
  for (int i = 0; i < MDMP_MAX_REQUESTS; i++) {
    if (mdmp_request_pool[i] != MPI_REQUEST_NULL) {
      MPI_Test(&mdmp_request_pool[i], &flag, MPI_STATUS_IGNORE);
    }
  }
}

void mdmp_init() {
  int provided;
  int is_initialised;
  MPI_Initialized(&is_initialised);

  if (!is_initialised) {
    MPI_Init(NULL, NULL);
    provided = MPI_THREAD_SINGLE;
    mdmp_owns_mpi = true;
  } else {
    MPI_Query_thread(&provided);
    mdmp_owns_mpi = false;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &global_my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &global_size);
  MPI_Comm_dup(MPI_COMM_WORLD, &mdmp_comm);

  // Reset tracking state cleanly
  mdmp_decl_req_counter = 0;
  mdmp_declarative_req_count = 0;
  mdmp_declarative_type_count = 0;
  decl_to_mpi_req.clear();

  for (int i = 0; i < MDMP_MAX_REQUESTS; ++i) mdmp_request_pool[i] = MPI_REQUEST_NULL;
  for (int i = 0; i < MDMP_MAX_DECLARATIVE_REQS; ++i) mdmp_declarative_requests[i] = MPI_REQUEST_NULL;
  
  const char* env_debug = getenv("MDMP_DEBUG");
  if (env_debug != NULL && atoi(env_debug) > 0) mdmp_debug_enabled = 1;

  mdmp_log("[MDMP Runtime] Starting up thread progress engine on rank %dn", global_my_rank);
  if (provided == MPI_THREAD_MULTIPLE) {
    mdmp_runtime_active = true;
    mdmp_progress_thread = std::thread(mdmp_progress_loop);
  } else {
    if (mdmp_get_rank() == 0) {
      printf("[MDMP Warning] MPI initialized without MPI_THREAD_MULTIPLE (Level: %d).\n", provided);
      printf("[MDMP Warning] Background progress engine DISABLED to prevent segfaults.\n");
    }
  }   
  mdmp_log("[MDMP Runtime] Initialised Rank %d / %d\n", global_my_rank, global_size);
}

void mdmp_final() {
  if (mdmp_runtime_active) {
    mdmp_runtime_active = false;
    if (mdmp_progress_thread.joinable()) mdmp_progress_thread.join();
  }

  for (int i = 0; i < mdmp_declarative_type_count; ++i) {
    MPI_Type_free(&mdmp_declarative_types_to_free[i]);
  }
  mdmp_declarative_type_count = 0;

  MPI_Comm_free(&mdmp_comm);
  if (mdmp_owns_mpi) MPI_Finalize();
}

int mdmp_get_rank() { return global_my_rank; }
int mdmp_get_size() { return global_size; }
double mdmp_wtime() { return MPI_Wtime(); }
void mdmp_sync()    { MPI_Barrier(mdmp_comm); }
void mdmp_set_debug(int enable) { mdmp_debug_enabled = enable; }
void mdmp_commregion_begin() {}

// --- NEW: The Bulletproof Safety Net ---
void mdmp_commregion_end() {
  if (mdmp_declarative_req_count > 0) {
    // If JIT waits missed anything inside deep control flow, catch it here cleanly
    MPI_Waitall(mdmp_declarative_req_count, mdmp_declarative_requests, MPI_STATUSES_IGNORE);
    for (int i = 0; i < mdmp_declarative_req_count; i++) mdmp_declarative_requests[i] = MPI_REQUEST_NULL;
  }
}

void mdmp_abort(int error_code) {
  fflush(stdout); 
  fprintf(stderr, "[MDMP FATAL] Rank %d called ABORT with error code %d. Terminating...\n", global_my_rank, error_code);
  fflush(stderr);
  MPI_Abort(mdmp_comm, error_code);
}

// ==============================================================================
// Updated Wait Router 
// ==============================================================================
void mdmp_wait(int req_id) {
  if (req_id == MDMP_PROCESS_NOT_INVOLVED) return;

  // 1. Specific Declarative Requests (Unbounded ID lookup)
  if (req_id >= MDMP_MAX_REQUESTS && req_id != MDMP_DECLARATIVE_WAIT) {
    int decl_idx = req_id - MDMP_MAX_REQUESTS;
    
    auto it = decl_to_mpi_req.find(decl_idx);
    if (it != decl_to_mpi_req.end()) {
      int mpi_idx = it->second;
      if (mpi_idx >= 0 && mdmp_declarative_requests[mpi_idx] != MPI_REQUEST_NULL) {
        MPI_Wait(&mdmp_declarative_requests[mpi_idx], MPI_STATUS_IGNORE);
        mdmp_declarative_requests[mpi_idx] = MPI_REQUEST_NULL;
      }
    }
  } 
  // 2. Imperative Ring Buffer Wait 
  else if (req_id >= 0 && req_id < MDMP_MAX_REQUESTS) {
    if (mdmp_request_pool[req_id] != MPI_REQUEST_NULL) {
      MPI_Wait(&mdmp_request_pool[req_id], MPI_STATUS_IGNORE);
    }
  }
  // 3. Monolithic Waitall (Legacy support)
  else if (req_id == MDMP_DECLARATIVE_WAIT) {
    if (mdmp_declarative_req_count > 0) {
      MPI_Waitall(mdmp_declarative_req_count, mdmp_declarative_requests, MPI_STATUSES_IGNORE);
      for(int i = 0; i < mdmp_declarative_req_count; i++) mdmp_declarative_requests[i] = MPI_REQUEST_NULL;
    }
    for (int i = 0; i < mdmp_declarative_type_count; ++i) MPI_Type_free(&mdmp_declarative_types_to_free[i]);
    mdmp_declarative_type_count = 0;
    mdmp_declarative_req_count = 0;
  }
  else {
    fprintf(stderr, "[MDMP Runtime Error] Invalid Request ID: %d\n", req_id);
    mdmp_abort(1);
  }
}

// ==============================================================================
// Imperative Calls 
// ==============================================================================
int mdmp_send(void* buf, size_t count, int type, size_t bytes, int sender, int dest, int tag) {
  if (dest < 0 || dest >= global_size || global_my_rank != sender) return MDMP_PROCESS_NOT_INVOLVED;
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active) lock.lock();
  uint32_t id = mdmp_req_counter % MDMP_MAX_REQUESTS;
  if (mdmp_request_pool[id] != MPI_REQUEST_NULL) { mdmp_abort(1); }    
  mdmp_req_counter++; 
  int actual_count = (type == 4) ? (int)bytes : (int)count;
  MPI_Isend(buf, actual_count, get_mpi_type(type), dest, tag, mdmp_comm, &mdmp_request_pool[id]);
  return (int)id;
}

int mdmp_recv(void* buf, size_t count, int type, size_t bytes, int receiver, int src, int tag) {
  if (src < 0 || src >= global_size || global_my_rank != receiver) return MDMP_PROCESS_NOT_INVOLVED;
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active) lock.lock();
  uint32_t id = mdmp_req_counter % MDMP_MAX_REQUESTS;
  if (mdmp_request_pool[id] != MPI_REQUEST_NULL) { mdmp_abort(1); }
  mdmp_req_counter++;
  int actual_count = (type == 4) ? (int)bytes : (int)count;
  MPI_Irecv(buf, actual_count, get_mpi_type(type), src, tag, mdmp_comm, &mdmp_request_pool[id]);
  return (int)id;
}

int mdmp_reduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes, int root_rank, int op) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active) lock.lock();
  uint32_t id = mdmp_req_counter % MDMP_MAX_REQUESTS;
  if (mdmp_request_pool[id] != MPI_REQUEST_NULL) { mdmp_abort(1); }
  mdmp_req_counter++;
  const void* final_sendbuf = (sendbuf == recvbuf && global_my_rank == root_rank) ? MPI_IN_PLACE : sendbuf;
  int actual_count = (type == 4) ? (int)bytes : (int)count;
  MPI_Ireduce(final_sendbuf, recvbuf, actual_count, get_mpi_type(type), get_mpi_op(op), root_rank, mdmp_comm, &mdmp_request_pool[id]);
  return (int)id;
}

int mdmp_gather(void* sendbuf, size_t sendcount, void* recvbuf, int type, size_t bytes,int root_rank) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active) lock.lock();
  uint32_t id = mdmp_req_counter % MDMP_MAX_REQUESTS;
  if (mdmp_request_pool[id] != MPI_REQUEST_NULL) { mdmp_abort(1); }
  mdmp_req_counter++;
  const void* final_sb = (sendbuf == recvbuf && global_my_rank == root_rank) ? MPI_IN_PLACE : sendbuf;
  int actual_count = (type == 4) ? (int)bytes : (int)sendcount;
  MPI_Igather(final_sb, actual_count, get_mpi_type(type), recvbuf, actual_count, get_mpi_type(type), root_rank, mdmp_comm, &mdmp_request_pool[id]);
  return (int)id;
}

int mdmp_allreduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes,int op) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active) lock.lock();
  uint32_t id = mdmp_req_counter % MDMP_MAX_REQUESTS;
  if (mdmp_request_pool[id] != MPI_REQUEST_NULL) { mdmp_abort(1); }
  mdmp_req_counter++;
  const void* final_sendbuf = (sendbuf == recvbuf) ? MPI_IN_PLACE : sendbuf;
  int actual_count = (type == 4) ? (int)bytes : (int)count;
  MPI_Iallreduce(final_sendbuf, recvbuf, actual_count, get_mpi_type(type), get_mpi_op(op), mdmp_comm, &mdmp_request_pool[id]);
  return (int)id;
}

int mdmp_allgather(void* sendbuf, size_t count, void* recvbuf, int type, size_t bytes) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active) lock.lock();
  uint32_t id = mdmp_req_counter % MDMP_MAX_REQUESTS;
  if (mdmp_request_pool[id] != MPI_REQUEST_NULL) { mdmp_abort(1); }
  mdmp_req_counter++;
  const void* final_sb = (sendbuf == recvbuf) ? MPI_IN_PLACE : sendbuf;
  int actual_count = (type == 4) ? (int)bytes : (int)count;
  MPI_Iallgather(final_sb, actual_count, get_mpi_type(type), recvbuf, actual_count, get_mpi_type(type), mdmp_comm, &mdmp_request_pool[id]);
  return (int)id;
}

int mdmp_bcast(void* buffer, size_t count, int type, size_t bytes, int root_rank) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active) lock.lock();
  uint32_t id = mdmp_req_counter % MDMP_MAX_REQUESTS;
  if (mdmp_request_pool[id] != MPI_REQUEST_NULL) { mdmp_abort(1); }
  mdmp_req_counter++;
  int actual_count = (type == 4) ? (int)bytes : (int)count;
  MPI_Ibcast(buffer, actual_count, get_mpi_type(type), root_rank, mdmp_comm, &mdmp_request_pool[id]);
  return (int)id;
}

// ==============================================================================
// Declarative Registration (Infinite IDs)
// ==============================================================================

int mdmp_register_send(void* buffer, size_t count, int type, size_t bytes, int sender_rank, int dest_rank, int tag) {
  if (dest_rank >= 0 && dest_rank < global_size && global_my_rank == sender_rank) {
    uint32_t id = mdmp_decl_req_counter++;
    send_queue.push_back({buffer, count, type, bytes, dest_rank, tag, (int)id}); 
    return (int)id + MDMP_MAX_REQUESTS;
  }
  return MDMP_PROCESS_NOT_INVOLVED;
}

int mdmp_register_recv(void* buffer, size_t count, int type, size_t bytes, int receiver_rank, int src_rank, int tag) {  
  if (src_rank >= 0 && src_rank < global_size && global_my_rank == receiver_rank) {
    uint32_t id = mdmp_decl_req_counter++;
    recv_queue.push_back({buffer, count, type, bytes, src_rank, tag, (int)id}); 
    return (int)id + MDMP_MAX_REQUESTS;
  }
  return MDMP_PROCESS_NOT_INVOLVED;
}

int mdmp_register_reduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes, int root_rank, int op) {
  uint32_t id = mdmp_decl_req_counter++;
  reduce_queue.push_back({sendbuf, recvbuf, count, type, bytes, root_rank, op, (int)id});
  return (int)id + MDMP_MAX_REQUESTS;
}

int mdmp_register_gather(void* sendbuf, size_t sendcount, void* recvbuf, int type, size_t bytes, int root_rank) {
  uint32_t id = mdmp_decl_req_counter++;
  gather_queue.push_back({sendbuf, sendcount, recvbuf, type, bytes, root_rank, (int)id});
  return (int)id + MDMP_MAX_REQUESTS;
}

int mdmp_register_allreduce(void* sendbuf, void* recvbuf, size_t count, int type,  size_t bytes, int op) {
  uint32_t id = mdmp_decl_req_counter++;
  allreduce_queue.push_back({sendbuf, recvbuf, count, type, bytes, op, (int)id});
  return (int)id + MDMP_MAX_REQUESTS;
}

int mdmp_register_allgather(void* sendbuf, size_t count, void* recvbuf, int type, size_t bytes) {
  uint32_t id = mdmp_decl_req_counter++;
  allgather_queue.push_back({sendbuf, count, recvbuf, type, bytes, (int)id});
  return (int)id + MDMP_MAX_REQUESTS;
}

int mdmp_register_bcast(void* buffer, size_t count, int type, size_t bytes, int root_rank) {
  uint32_t id = mdmp_decl_req_counter++;
  bcast_queue.push_back({buffer, count, type, bytes, root_rank, (int)id});
  return (int)id + MDMP_MAX_REQUESTS;
}
  
int mdmp_commit() {
  for (int i = 0; i < mdmp_declarative_type_count; ++i) {
    MPI_Type_free(&mdmp_declarative_types_to_free[i]);
  }
  mdmp_declarative_type_count = 0;
  mdmp_declarative_req_count = 0;
  
  // Safe to wipe tracking map for the new declarative batch
  decl_to_mpi_req.clear();

  static std::vector<int> blens_buffer;
  static std::vector<MPI_Aint> disps_buffer;
    
  auto ProcessQueue = [&](std::vector<RegisteredMsg>& queue, bool isSend) {     
    if (queue.empty()) return;

    std::stable_sort(queue.begin(), queue.end(), [](const RegisteredMsg& a, const RegisteredMsg& b) {
      if (a.rank != b.rank) return a.rank < b.rank;
      return a.tag < b.tag;
    });

    size_t i = 0;
    while (i < queue.size()) {
      int peer = queue[i].rank;
      int current_tag = queue[i].tag; 
      size_t j = i + 1;
            
      while (j < queue.size() && queue[j].rank == peer && queue[j].tag == current_tag) j++;

      size_t count = j - i;
      if (mdmp_declarative_req_count >= MDMP_MAX_DECLARATIVE_REQS) mdmp_abort(1);
            
      if (count == 1) { 
        int actual_count = (queue[i].type == 4) ? (int)queue[i].bytes : (int)queue[i].count;
        if (isSend) MPI_Isend(queue[i].buffer, actual_count, get_mpi_type(queue[i].type), peer, current_tag, mdmp_comm, &mdmp_declarative_requests[mdmp_declarative_req_count]);
        else        MPI_Irecv(queue[i].buffer, actual_count, get_mpi_type(queue[i].type), peer, current_tag, mdmp_comm, &mdmp_declarative_requests[mdmp_declarative_req_count]);
        
        decl_to_mpi_req[queue[i].req_id] = mdmp_declarative_req_count;
      } 
      else {
        blens_buffer.resize(count);
        disps_buffer.resize(count);
                
        for (size_t k = 0; k < count; k++) {
          int element_bytes = (int)queue[i + k].bytes;
          
          // --- NEW: Size-Inference Fallback ---
          // Protects against 0-byte transmission if the user's macro failed to pass sizeof(data)
          if (element_bytes == 0) {
	    int mpi_type_size;
	    MPI_Type_size(get_mpi_type(queue[i + k].type), &mpi_type_size);
	    element_bytes = (int)queue[i + k].count * mpi_type_size;
          }
          blens_buffer[k] = element_bytes;
          MPI_Get_address(queue[i + k].buffer, &disps_buffer[k]);
        }
        
        MPI_Datatype ntype;
        MPI_Type_create_hindexed((int)count, blens_buffer.data(), disps_buffer.data(), MPI_BYTE, &ntype);
        MPI_Type_commit(&ntype);
        mdmp_declarative_types_to_free[mdmp_declarative_type_count++] = ntype;

        if (isSend) MPI_Isend(MPI_BOTTOM, 1, ntype, peer, current_tag, mdmp_comm, &mdmp_declarative_requests[mdmp_declarative_req_count]);
        else        MPI_Irecv(MPI_BOTTOM, 1, ntype, peer, current_tag, mdmp_comm, &mdmp_declarative_requests[mdmp_declarative_req_count]);

        for (size_t k = 0; k < count; k++) {
	  decl_to_mpi_req[queue[i + k].req_id] = mdmp_declarative_req_count;
        }
      }
      
      mdmp_declarative_req_count++;
      i = j;
    }
  };
    
  ProcessQueue(recv_queue, false);
  ProcessQueue(send_queue, true);
     
  send_queue.clear();
  recv_queue.clear();

  for (auto& r : reduce_queue) {
    if (mdmp_declarative_req_count >= MDMP_MAX_DECLARATIVE_REQS) mdmp_abort(1);
    const void* final_sb = (r.sendbuf == r.recvbuf && global_my_rank == r.root) ? MPI_IN_PLACE : r.sendbuf;
    int actual_count = (r.type == 4) ? (int)r.bytes : (int)r.count;
    MPI_Ireduce(final_sb, r.recvbuf, actual_count, get_mpi_type(r.type), get_mpi_op(r.op), r.root, mdmp_comm, &mdmp_declarative_requests[mdmp_declarative_req_count]);
    decl_to_mpi_req[r.req_id] = mdmp_declarative_req_count++;
  }

  for (auto& g : gather_queue) {
    if (mdmp_declarative_req_count >= MDMP_MAX_DECLARATIVE_REQS) mdmp_abort(1);
    const void* final_sb = (g.sendbuf == g.recvbuf && global_my_rank == g.root) ? MPI_IN_PLACE : g.sendbuf;
    int actual_count = (g.type == 4) ? (int)g.bytes : (int)g.sendcount;
    MPI_Igather(final_sb, actual_count, get_mpi_type(g.type), g.recvbuf, actual_count, get_mpi_type(g.type), g.root, mdmp_comm, &mdmp_declarative_requests[mdmp_declarative_req_count]);
    decl_to_mpi_req[g.req_id] = mdmp_declarative_req_count++;
  }

  for (auto& ar : allreduce_queue) {
    if (mdmp_declarative_req_count >= MDMP_MAX_DECLARATIVE_REQS) mdmp_abort(1);
    const void* final_sb = (ar.sendbuf == ar.recvbuf) ? MPI_IN_PLACE : ar.sendbuf;
    int actual_count = (ar.type == 4) ? (int)ar.bytes : (int)ar.count;
    MPI_Iallreduce(final_sb, ar.recvbuf, actual_count, get_mpi_type(ar.type), get_mpi_op(ar.op), mdmp_comm, &mdmp_declarative_requests[mdmp_declarative_req_count]);
    decl_to_mpi_req[ar.req_id] = mdmp_declarative_req_count++;
  }

  for (auto& ag : allgather_queue) {
    if (mdmp_declarative_req_count >= MDMP_MAX_DECLARATIVE_REQS) mdmp_abort(1);
    const void* final_sb = (ag.sendbuf == ag.recvbuf) ? MPI_IN_PLACE : ag.sendbuf;
    int actual_count = (ag.type == 4) ? (int)ag.bytes : (int)ag.count;
    MPI_Iallgather(final_sb, actual_count, get_mpi_type(ag.type), ag.recvbuf, actual_count, get_mpi_type(ag.type), mdmp_comm, &mdmp_declarative_requests[mdmp_declarative_req_count]);
    decl_to_mpi_req[ag.req_id] = mdmp_declarative_req_count++;
  }

  for (auto& b : bcast_queue) {
    if (mdmp_declarative_req_count >= MDMP_MAX_DECLARATIVE_REQS) mdmp_abort(1);
    int actual_count = (b.type == 4) ? (int)b.bytes : (int)b.count;
    MPI_Ibcast(b.buffer, actual_count, get_mpi_type(b.type), b.root, mdmp_comm, &mdmp_declarative_requests[mdmp_declarative_req_count]);
    decl_to_mpi_req[b.req_id] = mdmp_declarative_req_count++;
  }

  reduce_queue.clear();
  gather_queue.clear();
  allreduce_queue.clear();
  allgather_queue.clear();
  bcast_queue.clear();
    
  return MDMP_DECLARATIVE_WAIT; 
}
