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
  // If the user accidentally passed a raw MPI macro (which are massive hex values), 
  // safely cast it and use it directly to prevent defaulting to SUM!
  if (op_code > 10) return (MPI_Op)(intptr_t)op_code;
    
  switch(op_code) {
  case 0: return MPI_SUM;
  case 1: return MPI_MAX;
  case 2: return MPI_MIN;
  case 3: return MPI_PROD;
  default: return MPI_SUM;
  }
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

// ==============================================================================
// Background progress engine
// ==============================================================================
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

void mdmp_init() {
  int provided;
  int is_initialised;

  MPI_Initialized(&is_initialised);

  if (!is_initialised) {
    // MPI is not running. We take control and demand multithreading.
    //        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    MPI_Init(NULL, NULL);
    provided = MPI_THREAD_SINGLE;
    mdmp_owns_mpi = true;
  } else {
    // The test suite already started MPI. Query what thread level we actually have.
    MPI_Query_thread(&provided);
    mdmp_owns_mpi = false;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &global_my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &global_size);
    
  MPI_Comm_dup(MPI_COMM_WORLD, &mdmp_comm);

  for (int i = 0; i < MDMP_MAX_REQUESTS; ++i) {
    mdmp_request_pool[i] = MPI_REQUEST_NULL;
  }
  
  // Read the environment variable once at startup
  const char* env_debug = getenv("MDMP_DEBUG");
  if (env_debug != NULL && atoi(env_debug) > 0) {
    mdmp_debug_enabled = 1;
  }

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
  // Gracefully spin down the progress engine
  if (mdmp_runtime_active) {
    mdmp_runtime_active = false;
    if (mdmp_progress_thread.joinable()) {
      mdmp_progress_thread.join();
    }
    mdmp_log("[MDMP Runtime] Shut down thread progress engine on rank %dn", global_my_rank);
  }
    
  // Free our private communicator before shutting down MPI
  MPI_Comm_free(&mdmp_comm);

  mdmp_log("[MDMP Runtime] Finalised Rank %d / %d\n", global_my_rank, global_size);
   
  // Only shut down MPI if we were the ones who started it
  if (mdmp_owns_mpi) {
    MPI_Finalize();
  }

}

int mdmp_get_rank() { return global_my_rank; }
int mdmp_get_size() { return global_size; }
double mdmp_wtime() { return MPI_Wtime(); }
void mdmp_sync()    { MPI_Barrier(mdmp_comm); }

void mdmp_set_debug(int enable) {
  mdmp_debug_enabled = enable;
}

void mdmp_commregion_begin() {}
void mdmp_commregion_end() {}


void mdmp_abort(int error_code) {
  fflush(stdout); 
  fprintf(stderr, "[MDMP FATAL] Rank %d called ABORT with error code %d. Terminating...\n", global_my_rank, error_code);
  fflush(stderr);
  MPI_Abort(mdmp_comm, error_code);
}

void mdmp_wait(int req_id) {
    if (req_id >= 0 && req_id < MDMP_MAX_REQUESTS) {
        
        // Only wait if there is a real request sitting there
        if (mdmp_request_pool[req_id] != MPI_REQUEST_NULL) {
            MPI_Wait(&mdmp_request_pool[req_id], MPI_STATUS_IGNORE);
            // MPI_Wait automatically sets mdmp_request_pool[req_id] = MPI_REQUEST_NULL here
        }
        
    } else {
        fprintf(stderr, "[MDMP Runtime Error] Invalid Request ID: %d\n", req_id);
	mdmp_abort(1);
    }
}

int mdmp_send(void* buf, size_t c, int t, size_t bytes, int s, int d, int tag) {
  if (d < 0 || d >= global_size || global_my_rank != s) return -2;
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active) lock.lock();
  uint32_t id = mdmp_req_counter % MDMP_MAX_REQUESTS;
    
  if (mdmp_request_pool[id] != MPI_REQUEST_NULL) {
    fprintf(stderr, "[MDMP FATAL] Request Ring Buffer Overflow! Exceeded %d concurrent active requests.\n", MDMP_MAX_REQUESTS);
    mdmp_abort(1);
  }    
  mdmp_req_counter++; 
    
  // If it's a custom struct (MPI_BYTE), the true count is the total bytes
  int actual_count = (t == 4) ? (int)bytes : (int)c;
    
  MPI_Isend(buf, actual_count, get_mpi_type(t), d, tag, mdmp_comm, &mdmp_request_pool[id]);
  return (int)id;
}

int mdmp_recv(void* buf, size_t c, int t, size_t bytes, int r, int s, int tag) {
  if (s < 0 || s >= global_size || global_my_rank != r) return -2;
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active) lock.lock();
  uint32_t id = mdmp_req_counter % MDMP_MAX_REQUESTS;

  if (mdmp_request_pool[id] != MPI_REQUEST_NULL) {
    fprintf(stderr, "[MDMP FATAL] Request Ring Buffer Overflow! Exceeded %d concurrent active requests.\n", MDMP_MAX_REQUESTS);
    mdmp_abort(1);
  }
  mdmp_req_counter++;
  
  int actual_count = (t == 4) ? (int)bytes : (int)c;
    
  MPI_Irecv(buf, actual_count, get_mpi_type(t), s, tag, mdmp_comm, &mdmp_request_pool[id]);
  return (int)id;
}

// Collectives
int mdmp_reduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes, int root, int op) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active) lock.lock();

  uint32_t id = mdmp_req_counter % MDMP_MAX_REQUESTS;
  if (mdmp_request_pool[id] != MPI_REQUEST_NULL) {
    fprintf(stderr, "[MDMP FATAL] Request Ring Buffer Overflow! Exceeded %d concurrent active requests.\n", MDMP_MAX_REQUESTS);
    mdmp_abort(1);
  }
  mdmp_req_counter++;
  
  const void* final_sendbuf = sendbuf;
  if (sendbuf == recvbuf && global_my_rank == root) final_sendbuf = MPI_IN_PLACE;

  int actual_count = (type == 4) ? (int)bytes : (int)count;
  MPI_Ireduce(final_sendbuf, recvbuf, actual_count, get_mpi_type(type), get_mpi_op(op), root, mdmp_comm, &mdmp_request_pool[id]);
  return (int)id;
}

int mdmp_gather(void* sb, size_t sc, void* rb, int t, size_t bytes, int root) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active) lock.lock();

  uint32_t id = mdmp_req_counter % MDMP_MAX_REQUESTS;
  if (mdmp_request_pool[id] != MPI_REQUEST_NULL) {
    fprintf(stderr, "[MDMP FATAL] Request Ring Buffer Overflow! Exceeded %d concurrent active requests.\n", MDMP_MAX_REQUESTS);
    mdmp_abort(1);
  }
  mdmp_req_counter++;
  
  const void* final_sb = sb;
  if (sb == rb && global_my_rank == root) final_sb = MPI_IN_PLACE;

  int actual_count = (t == 4) ? (int)bytes : (int)sc;
  MPI_Igather(final_sb, actual_count, get_mpi_type(t), rb, actual_count, get_mpi_type(t), root, mdmp_comm, &mdmp_request_pool[id]);
  return (int)id;
}

int mdmp_allreduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes, int op) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active) lock.lock();

  uint32_t id = mdmp_req_counter % MDMP_MAX_REQUESTS;
  if (mdmp_request_pool[id] != MPI_REQUEST_NULL) {
    fprintf(stderr, "[MDMP FATAL] Request Ring Buffer Overflow! Exceeded %d concurrent active requests.\n", MDMP_MAX_REQUESTS);
    mdmp_abort(1);
  }
  mdmp_req_counter++;
  
  const void* final_sendbuf = sendbuf;
  if (sendbuf == recvbuf) final_sendbuf = MPI_IN_PLACE;
  
  int actual_count = (type == 4) ? (int)bytes : (int)count;
    
  MPI_Op mpi_op = get_mpi_op(op); 
    
  MPI_Iallreduce(final_sendbuf, recvbuf, actual_count, get_mpi_type(type), mpi_op, mdmp_comm, &mdmp_request_pool[id]);
  return (int)id;
}

int mdmp_allgather(void* sb, size_t c, void* rb, int t, size_t bytes) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active) lock.lock();

  uint32_t id = mdmp_req_counter % MDMP_MAX_REQUESTS;
  if (mdmp_request_pool[id] != MPI_REQUEST_NULL) {
    fprintf(stderr, "[MDMP FATAL] Request Ring Buffer Overflow! Exceeded %d concurrent active requests.\n", MDMP_MAX_REQUESTS);
    mdmp_abort(1);
  }
  mdmp_req_counter++;

  const void* final_sb = sb;
  if (sb == rb) final_sb = MPI_IN_PLACE;

  int actual_count = (t == 4) ? (int)bytes : (int)c;
  MPI_Iallgather(final_sb, actual_count, get_mpi_type(t), rb, actual_count, get_mpi_type(t), mdmp_comm, &mdmp_request_pool[id]);
  return (int)id;
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

  static std::vector<int> blens_buffer;
  static std::vector<MPI_Aint> disps_buffer;
    
    
    
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

      // Get a request id
      uint32_t id = mdmp_req_counter % MDMP_MAX_REQUESTS;
      if (mdmp_request_pool[id] != MPI_REQUEST_NULL) {
	fprintf(stderr, "[MDMP FATAL] Request Ring Buffer Overflow!\n");
	mdmp_abort(1);
      }
      mdmp_req_counter++;
            
      // Fire individually if there is only 1 message
      if (count == 1) { 

	int actual_count = (queue[i].type == 4) ? (int)queue[i].bytes : (int)queue[i].count;
                
	if (isSend) MPI_Isend(queue[i].buffer, actual_count, get_mpi_type(queue[i].type), peer, current_tag, mdmp_comm, &mdmp_request_pool[id]);
	else        MPI_Irecv(queue[i].buffer, actual_count, get_mpi_type(queue[i].type), peer, current_tag, mdmp_comm, &mdmp_request_pool[id]);
      } 
      // 4. Fire the Zero-Copy Hardware Datatype if there are multiple
      else {

	blens_buffer.resize(count);
	disps_buffer.resize(count);
                
	for (size_t k = 0; k < count; k++) {
	  blens_buffer[k] = (int)queue[i + k].bytes;
	  MPI_Get_address(queue[i + k].buffer, &disps_buffer[k]);
	}
        
	MPI_Datatype ntype;
	MPI_Type_create_hindexed((int)count, blens_buffer.data(), disps_buffer.data(), MPI_BYTE, &ntype);
	MPI_Type_commit(&ntype);
	custom_types_to_free.push_back(ntype); 

	if (isSend) MPI_Isend(MPI_BOTTOM, 1, ntype, peer, current_tag, mdmp_comm, &mdmp_request_pool[id]);
	else        MPI_Irecv(MPI_BOTTOM, 1, ntype, peer, current_tag, mdmp_comm, &mdmp_request_pool[id]);
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
