#include "mdmp_runtime.h"

static int global_my_rank = -1;
static int global_size = -1;
static int mdmp_debug_enabled = 0;

MPI_Comm mdmp_comm; 
std::atomic<bool> mdmp_runtime_active{false};
std::thread mdmp_progress_thread;
std::mutex mdmp_mpi_mutex;
bool mdmp_owns_mpi = false;

static constexpr int MDMP_DECL_BATCH_TOKEN_BASE = (1 << 30);

struct DeclWaitTarget {
  uint32_t BatchSerial;
  int MPIIndex;
};

struct DeclarativeBatch {
  uint32_t Serial = 0;
  std::vector<MPI_Request> Requests;
  std::vector<MPI_Datatype> TypesToFree;
  std::vector<int> LogicalReqIDs; // raw logical IDs, not offset by MDMP_MAX_REQUESTS
};

MPI_Request mdmp_request_pool[MDMP_MAX_REQUESTS];
uint32_t mdmp_imper_req_counter = 0;

uint32_t mdmp_decl_req_counter = 0;    // logical registration ids
uint32_t mdmp_decl_batch_counter = 0;  // commit batch ids

std::unordered_map<uint32_t, DeclarativeBatch> mdmp_active_decl_batches;
std::unordered_map<int, DeclWaitTarget> decl_to_req_target;

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


inline MPI_Datatype get_mpi_type(int type_code) {
  switch(type_code) {
  case 0: return MPI_INT; case 1: return MPI_DOUBLE; case 2: return MPI_FLOAT;
  case 3: return MPI_CHAR; case 4: return MPI_BYTE; default: return MPI_BYTE;
  }
}

inline MPI_Op get_mpi_op(int op_code) {
  if (op_code > 10) return (MPI_Op)(intptr_t)op_code;
  switch(op_code) {
  case 0: return MPI_SUM; case 1: return MPI_MAX; case 2: return MPI_MIN;
  case 3: return MPI_PROD; default: return MPI_SUM;
  }
}

inline int mdmp_actual_count(size_t count, int type, size_t bytes) {
  return (type == 4) ? (int)bytes : (int)count;
}

int mdmp_alloc_imperative_slot_unlocked() {
  uint32_t id = mdmp_imper_req_counter % MDMP_MAX_REQUESTS;

  if (mdmp_request_pool[id] != MPI_REQUEST_NULL) {
    fprintf(stderr,
            "[MDMP FATAL] Request ring buffer overflow! "
            "Exceeded %d concurrent imperative requests.\n",
            MDMP_MAX_REQUESTS);
    mdmp_abort(1);
  }

  mdmp_imper_req_counter++;
  return (int)id;
}


inline bool mdmp_is_decl_batch_token(int token) {
  return token >= MDMP_DECL_BATCH_TOKEN_BASE;
}

inline int mdmp_make_decl_batch_token(uint32_t serial) {
  return MDMP_DECL_BATCH_TOKEN_BASE + (int)serial;
}

inline uint32_t mdmp_decl_batch_serial_from_token(int token) {
  return (uint32_t)(token - MDMP_DECL_BATCH_TOKEN_BASE);
}

bool mdmp_batch_all_complete(const DeclarativeBatch &B) {
  for (const MPI_Request &R : B.Requests) {
    if (R != MPI_REQUEST_NULL)
      return false;
  }
  return true;
}

void mdmp_retire_declarative_batch(uint32_t serial) {
  auto It = mdmp_active_decl_batches.find(serial);
  if (It == mdmp_active_decl_batches.end())
    return;

  DeclarativeBatch &B = It->second;

  for (int LogicalID : B.LogicalReqIDs) {
    decl_to_req_target.erase(LogicalID);
  }

  for (MPI_Datatype &Ty : B.TypesToFree) {
    if (Ty != MPI_DATATYPE_NULL) {
      MPI_Type_free(&Ty);
    }
  }

  mdmp_active_decl_batches.erase(It);
}

void mdmp_wait_declarative_batch_unlocked(uint32_t serial) {
  auto It = mdmp_active_decl_batches.find(serial);
  if (It == mdmp_active_decl_batches.end())
    return;

  DeclarativeBatch &B = It->second;

  if (!B.Requests.empty()) {
    mdmp_check_mpi(
        MPI_Waitall((int)B.Requests.size(), B.Requests.data(), MPI_STATUSES_IGNORE),
        "MPI_Waitall(declarative batch)");
  }

  mdmp_retire_declarative_batch(serial);
}

void mdmp_wait_all_declarative_batches_unlocked() {
  std::vector<uint32_t> Serials;
  Serials.reserve(mdmp_active_decl_batches.size());

  for (auto &KV : mdmp_active_decl_batches) {
    Serials.push_back(KV.first);
  }

  for (uint32_t Serial : Serials) {
    mdmp_wait_declarative_batch_unlocked(Serial);
  }
}

bool mdmp_any_declarative_batches_active() {
  return !mdmp_active_decl_batches.empty();
}

void mdmp_check_mpi(int rc, const char *where) {
  if (rc == MPI_SUCCESS) return;

  char errstr[MPI_MAX_ERROR_STRING];
  int len = 0;
  MPI_Error_string(rc, errstr, &len);

  fprintf(stderr,
          "[MDMP FATAL] Rank %d: %s failed with MPI error: %.*s\n",
          global_my_rank, where, len, errstr);
  mdmp_abort(rc);
}

bool mdmp_has_active_requests() {
  for (int i = 0; i < MDMP_MAX_REQUESTS; ++i) {
    if (mdmp_request_pool[i] != MPI_REQUEST_NULL)
      return true;
  }

  for (auto &KV : mdmp_active_decl_batches) {
    DeclarativeBatch &B = KV.second;
    for (MPI_Request &R : B.Requests) {
      if (R != MPI_REQUEST_NULL)
        return true;
    }
  }

  return false;
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
  for (auto &KV : mdmp_active_decl_batches) {
    DeclarativeBatch &B = KV.second;
    for (MPI_Request &R : B.Requests) {
      if (R != MPI_REQUEST_NULL) {
        MPI_Test(&R, &flag, MPI_STATUS_IGNORE);
      }
    }
  }
}

void mdmp_maybe_progress() {
  if (mdmp_runtime_active)
    return;

  if (!mdmp_has_active_requests())
    return;

  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active)
    return;

  mdmp_progress();
}

// Deduplicate IDs while preserving first-seen order.
// Skips MDMP_PROCESS_NOT_INVOLVED.
int mdmp_dedup_ids_linear(const int *ids, int n, int *uniq, int cap) {
  int u = 0;

  for (int i = 0; i < n; ++i) {
    int id = ids[i];
    if (id == MDMP_PROCESS_NOT_INVOLVED)
      continue;

    bool seen = false;
    for (int j = 0; j < u; ++j) {
      if (uniq[j] == id) {
        seen = true;
        break;
      }
    }

    if (!seen) {
      if (u >= cap) {
        fprintf(stderr,
                "[MDMP FATAL] mdmp_dedup_ids_linear capacity exceeded "
                "(cap=%d, n=%d)\n",
                cap, n);
        mdmp_abort(1);
      }
      uniq[u++] = id;
    }
  }

  return u;
}

void mdmp_wait_many_sequential_unlocked(const int *uniq, int count) {
  for (int i = 0; i < count; ++i) {
    mdmp_wait_unlocked(uniq[i]);
  }
}

// Stack-only imperative batch path for moderate-size imperative-only groups.
void mdmp_wait_many_imperative_batch_stack_unlocked(const int *uniq, int count) {
  if (count < 0 || count > MDMP_WAIT_MANY_STACK_CUTOFF) {
    fprintf(stderr,
            "[MDMP Runtime Error] Invalid stack batch count in "
            "mdmp_wait_many_imperative_batch_stack_unlocked: %d\n",
            count);
    mdmp_abort(1);
  }
  
  MPI_Request batch[MDMP_WAIT_MANY_STACK_CUTOFF];
  int slots[MDMP_WAIT_MANY_STACK_CUTOFF];
  int active = 0;

  for (int i = 0; i < count; ++i) {
    int id = uniq[i];

    if (id < 0 || id >= MDMP_MAX_REQUESTS) {
      fprintf(stderr,
              "[MDMP Runtime Error] Invalid imperative request ID in "
              "mdmp_wait_many stack batch: %d\n",
              id);
      mdmp_abort(1);
    }

    if (mdmp_request_pool[id] != MPI_REQUEST_NULL) {
      slots[active] = id;
      batch[active] = mdmp_request_pool[id];
      ++active;
    }
  }

  if (active == 0)
    return;

  if (active == 1) {
    mdmp_wait_unlocked(slots[0]);
    return;
  }

  mdmp_check_mpi(MPI_Waitall(active, batch, MPI_STATUSES_IGNORE), "MPI_Waitall(imperative stack batch)");

  for (int i = 0; i < active; ++i) {
    mdmp_request_pool[slots[i]] = MPI_REQUEST_NULL;
  }
}

void mdmp_wait_unlocked(int req_id) {
  if (req_id == MDMP_PROCESS_NOT_INVOLVED)
    return;

  // 1. Whole declarative batch token
  if (mdmp_is_decl_batch_token(req_id)) {
    uint32_t batchSerial = mdmp_decl_batch_serial_from_token(req_id);
    mdmp_wait_declarative_batch_unlocked(batchSerial);
    return;
  }

  // 2. Specific declarative logical request
  if (req_id >= MDMP_MAX_REQUESTS && req_id != MDMP_DECLARATIVE_WAIT) {
    int logicalReqID = req_id - MDMP_MAX_REQUESTS;

    auto it = decl_to_req_target.find(logicalReqID);
    if (it != decl_to_req_target.end()) {
      uint32_t batchSerial = it->second.BatchSerial;
      int mpi_idx = it->second.MPIIndex;

      auto bit = mdmp_active_decl_batches.find(batchSerial);
      if (bit != mdmp_active_decl_batches.end()) {
        DeclarativeBatch &B = bit->second;

        if (mpi_idx >= 0 &&
            mpi_idx < (int)B.Requests.size() &&
            B.Requests[mpi_idx] != MPI_REQUEST_NULL) {
          mdmp_check_mpi(
              MPI_Wait(&B.Requests[mpi_idx], MPI_STATUS_IGNORE),
              "MPI_Wait(declarative logical)");
          B.Requests[mpi_idx] = MPI_REQUEST_NULL;
        }

        if (mdmp_batch_all_complete(B)) {
          mdmp_retire_declarative_batch(batchSerial);
        }
      }
    } else if (mdmp_debug_enabled) {
      fprintf(stderr,
              "[MDMP Warning] Rank %d: unknown declarative request ID %d\n",
              global_my_rank, req_id);
    }
    return;
  }

  // 3. Imperative request
  if (req_id >= 0 && req_id < MDMP_MAX_REQUESTS) {
    if (mdmp_request_pool[req_id] != MPI_REQUEST_NULL) {
      mdmp_check_mpi(
          MPI_Wait(&mdmp_request_pool[req_id], MPI_STATUS_IGNORE),
          "MPI_Wait(imperative)");
      mdmp_request_pool[req_id] = MPI_REQUEST_NULL;
    }
    return;
  }

  // 4. Legacy: wait all active declarative batches
  if (req_id == MDMP_DECLARATIVE_WAIT) {
    mdmp_wait_all_declarative_batches_unlocked();
    return;
  }

  fprintf(stderr, "[MDMP Runtime Error] Invalid Request ID: %d\n", req_id);
  mdmp_abort(1);
}

void mdmp_wait(int req_id) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active)
    lock.lock();

  mdmp_wait_unlocked(req_id);
}

void mdmp_wait_many(const int *ids, int n) {
  if (ids == nullptr || n <= 0)
    return;

  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active)
    lock.lock();

  // ------------------------------------------------------------
  // Tiny hot-path: fixed-array dedup + sequential waits
  // ------------------------------------------------------------
  if (n <= MDMP_WAIT_MANY_TINY_CUTOFF) {
    int uniq[MDMP_WAIT_MANY_TINY_CUTOFF];
    int u = mdmp_dedup_ids_linear(ids, n, uniq, MDMP_WAIT_MANY_TINY_CUTOFF);
    mdmp_wait_many_sequential_unlocked(uniq, u);
    return;
  }

  // ------------------------------------------------------------
  // Classify the batch
  // ------------------------------------------------------------
  bool imperative_only = true;
  bool declarative_batch_tokens_only = true;

  for (int i = 0; i < n; ++i) {
    int id = ids[i];
    if (id == MDMP_PROCESS_NOT_INVOLVED)
      continue;

    if (!(id >= 0 && id < MDMP_MAX_REQUESTS))
      imperative_only = false;

    if (!mdmp_is_decl_batch_token(id))
      declarative_batch_tokens_only = false;
  }

  // ------------------------------------------------------------
  // Fast path for declarative batch tokens only
  // (compiler should mostly emit these after mdmp_commit()).
  // ------------------------------------------------------------
  if (declarative_batch_tokens_only) {
    if (n <= MDMP_WAIT_MANY_STACK_CUTOFF) {
      int uniq[MDMP_WAIT_MANY_STACK_CUTOFF];
      int u = mdmp_dedup_ids_linear(ids, n, uniq, MDMP_WAIT_MANY_STACK_CUTOFF);
      mdmp_wait_many_sequential_unlocked(uniq, u);
      return;
    }

    std::vector<int> uniq;
    uniq.reserve(n);

    for (int i = 0; i < n; ++i) {
      int id = ids[i];
      if (id == MDMP_PROCESS_NOT_INVOLVED)
        continue;

      bool seen = false;
      for (int x : uniq) {
        if (x == id) {
          seen = true;
          break;
        }
      }

      if (!seen)
        uniq.push_back(id);
    }

    for (int id : uniq)
      mdmp_wait_unlocked(id);

    return;
  }

  // ------------------------------------------------------------
  // Mixed or declarative-containing batch:
  // keep it simple and cheap, preserving semantics.
  // ------------------------------------------------------------
  if (!imperative_only) {
    if (n <= MDMP_WAIT_MANY_STACK_CUTOFF) {
      int uniq[MDMP_WAIT_MANY_STACK_CUTOFF];
      int u = mdmp_dedup_ids_linear(ids, n, uniq, MDMP_WAIT_MANY_STACK_CUTOFF);
      mdmp_wait_many_sequential_unlocked(uniq, u);
      return;
    }

    // Rare fallback path for large mixed/declarative batches.
    std::vector<int> uniq;
    uniq.reserve(n);

    for (int i = 0; i < n; ++i) {
      int id = ids[i];
      if (id == MDMP_PROCESS_NOT_INVOLVED)
        continue;

      bool seen = false;
      for (int x : uniq) {
        if (x == id) {
          seen = true;
          break;
        }
      }

      if (!seen)
        uniq.push_back(id);
    }

    for (int id : uniq)
      mdmp_wait_unlocked(id);

    return;
  }

  // ------------------------------------------------------------
  // Imperative-only batch
  // ------------------------------------------------------------

  // Medium-size imperative-only batch: stack dedup.
  if (n <= MDMP_WAIT_MANY_STACK_CUTOFF) {
    int uniq[MDMP_WAIT_MANY_STACK_CUTOFF];
    int u = mdmp_dedup_ids_linear(ids, n, uniq, MDMP_WAIT_MANY_STACK_CUTOFF);

    // For small unique counts, sequential is cheaper than setting up Waitall.
    if (u < MDMP_WAIT_MANY_MPI_BATCH_MIN) {
      mdmp_wait_many_sequential_unlocked(uniq, u);
      return;
    }

    mdmp_wait_many_imperative_batch_stack_unlocked(uniq, u);
    return;
  }

  // Large imperative-only batch: dedup via bitmap, then MPI_Waitall.
  std::vector<unsigned char> seen(MDMP_MAX_REQUESTS, 0);
  std::vector<int> uniq;
  uniq.reserve((n < MDMP_MAX_REQUESTS) ? n : MDMP_MAX_REQUESTS);

  for (int i = 0; i < n; ++i) {
    int id = ids[i];
    if (id == MDMP_PROCESS_NOT_INVOLVED)
      continue;

    if (!seen[id]) {
      seen[id] = 1;
      uniq.push_back(id);
    }
  }

  if ((int)uniq.size() < MDMP_WAIT_MANY_MPI_BATCH_MIN) {
    for (int id : uniq)
      mdmp_wait_unlocked(id);
    return;
  }

  std::vector<MPI_Request> batch;
  std::vector<int> activeSlots;
  batch.reserve(uniq.size());
  activeSlots.reserve(uniq.size());

  for (int id : uniq) {
    if (mdmp_request_pool[id] != MPI_REQUEST_NULL) {
      activeSlots.push_back(id);
      batch.push_back(mdmp_request_pool[id]);
    }
  }

  if (batch.empty())
    return;

  if (batch.size() == 1) {
    mdmp_wait_unlocked(activeSlots[0]);
    return;
  }

  mdmp_check_mpi(
      MPI_Waitall((int)batch.size(), batch.data(), MPI_STATUSES_IGNORE),
      "MPI_Waitall(imperative heap batch)");

  for (int id : activeSlots) {
    mdmp_request_pool[id] = MPI_REQUEST_NULL;
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

  mdmp_imper_req_counter = 0;
  mdmp_decl_req_counter = 0;
  mdmp_decl_batch_counter = 0;
  mdmp_active_decl_batches.clear();
  decl_to_req_target.clear();
  
  for (int i = 0; i < MDMP_MAX_REQUESTS; ++i) mdmp_request_pool[i] = MPI_REQUEST_NULL;
  
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

  mdmp_wait_all_declarative_batches_unlocked();

  MPI_Comm_free(&mdmp_comm);
  if (mdmp_owns_mpi) MPI_Finalize();
}

int mdmp_get_rank() { return global_my_rank; }
int mdmp_get_size() { return global_size; }
double mdmp_wtime() { return MPI_Wtime(); }
void mdmp_sync()    { MPI_Barrier(mdmp_comm); }
void mdmp_set_debug(int enable) { mdmp_debug_enabled = enable; }
void mdmp_commregion_begin() {}

void mdmp_commregion_end() {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active)
    lock.lock();

  mdmp_wait_all_declarative_batches_unlocked();
}

void mdmp_abort(int error_code) {
  fflush(stdout); 
  fprintf(stderr, "[MDMP FATAL] Rank %d called ABORT with error code %d. Terminating...\n", global_my_rank, error_code);
  fflush(stderr);
  MPI_Abort(mdmp_comm, error_code);
}

// ==============================================================================
// Imperative Calls 
// ==============================================================================
int mdmp_send(void* buf, size_t count, int type, size_t bytes, int sender, int dest, int tag) {
  if (dest < 0 || dest >= global_size || global_my_rank != sender)
    return MDMP_PROCESS_NOT_INVOLVED;

  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active)
    lock.lock();

  int id = mdmp_alloc_imperative_slot_unlocked();
  int actual_count = mdmp_actual_count(count, type, bytes);

  mdmp_check_mpi(MPI_Isend(buf, actual_count, get_mpi_type(type), dest, tag, mdmp_comm, &mdmp_request_pool[id]), "MPI_Isend");

  return id;
}

int mdmp_recv(void* buf, size_t count, int type, size_t bytes, int receiver, int src, int tag) {
  if (src < 0 || src >= global_size || global_my_rank != receiver)
    return MDMP_PROCESS_NOT_INVOLVED;

  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active)
    lock.lock();

  int id = mdmp_alloc_imperative_slot_unlocked();
  int actual_count = mdmp_actual_count(count, type, bytes);

  mdmp_check_mpi(MPI_Irecv(buf, actual_count, get_mpi_type(type), src, tag, mdmp_comm, &mdmp_request_pool[id]), "MPI_Irecv");

  return id;
}

int mdmp_reduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes, int root_rank, int op) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active)
    lock.lock();

  int id = mdmp_alloc_imperative_slot_unlocked();

  const void* final_sendbuf = (sendbuf == recvbuf && global_my_rank == root_rank) ? MPI_IN_PLACE : sendbuf;

  int actual_count = mdmp_actual_count(count, type, bytes);

  mdmp_check_mpi(MPI_Ireduce(final_sendbuf, recvbuf, actual_count, get_mpi_type(type), get_mpi_op(op), root_rank, mdmp_comm, &mdmp_request_pool[id]), "MPI_Ireduce");

  return id;
}

int mdmp_gather(void* sendbuf, size_t sendcount, void* recvbuf, int type, size_t bytes, int root_rank) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active)
    lock.lock();

  int id = mdmp_alloc_imperative_slot_unlocked();

  const void* final_sb = (sendbuf == recvbuf && global_my_rank == root_rank) ? MPI_IN_PLACE : sendbuf;

  int actual_count = mdmp_actual_count(sendcount, type, bytes);

  mdmp_check_mpi(MPI_Igather(final_sb, actual_count, get_mpi_type(type), recvbuf, actual_count, get_mpi_type(type), root_rank, mdmp_comm, &mdmp_request_pool[id]), "MPI_Igather");

  return id;
}

int mdmp_allreduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes, int op) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active)
    lock.lock();

  int id = mdmp_alloc_imperative_slot_unlocked();

  const void* final_sendbuf = (sendbuf == recvbuf) ? MPI_IN_PLACE : sendbuf;
      
  int actual_count = mdmp_actual_count(count, type, bytes);

  mdmp_check_mpi(MPI_Iallreduce(final_sendbuf, recvbuf, actual_count, get_mpi_type(type), get_mpi_op(op), mdmp_comm, &mdmp_request_pool[id]), "MPI_Iallreduce");

  return id;
}

int mdmp_allgather(void* sendbuf, size_t count, void* recvbuf, int type, size_t bytes) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active)
    lock.lock();

  int id = mdmp_alloc_imperative_slot_unlocked();

  const void* final_sb = (sendbuf == recvbuf) ? MPI_IN_PLACE : sendbuf;

  int actual_count = mdmp_actual_count(count, type, bytes);

  mdmp_check_mpi(MPI_Iallgather(final_sb, actual_count, get_mpi_type(type), recvbuf, actual_count, get_mpi_type(type), mdmp_comm, &mdmp_request_pool[id]), "MPI_Iallgather");

  return id;
}

int mdmp_bcast(void* buffer, size_t count, int type, size_t bytes, int root_rank) {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active)
    lock.lock();

  int id = mdmp_alloc_imperative_slot_unlocked();
  int actual_count = mdmp_actual_count(count, type, bytes);

  mdmp_check_mpi(MPI_Ibcast(buffer, actual_count, get_mpi_type(type), root_rank, mdmp_comm, &mdmp_request_pool[id]), "MPI_Ibcast");

  return id;
}

// ==============================================================================
// Declarative Registration (Infinite IDs)
// ==============================================================================

int mdmp_register_send(void* buffer, size_t count, int type, size_t bytes, int sender_rank, int dest_rank, int tag) {
  if (dest_rank >= 0 && dest_rank < global_size && global_my_rank == sender_rank) {
    if ((int)mdmp_decl_req_counter + MDMP_MAX_REQUESTS >= MDMP_DECL_BATCH_TOKEN_BASE) {
      fprintf(stderr, "[MDMP FATAL] Declarative logical ID space exhausted.\n");
      mdmp_abort(1);
    }    
    uint32_t id = mdmp_decl_req_counter++;
    send_queue.push_back({buffer, count, type, bytes, dest_rank, tag, (int)id}); 
    return (int)id + MDMP_MAX_REQUESTS;
  }
  return MDMP_PROCESS_NOT_INVOLVED;
}

int mdmp_register_recv(void* buffer, size_t count, int type, size_t bytes, int receiver_rank, int src_rank, int tag) {  
  if (src_rank >= 0 && src_rank < global_size && global_my_rank == receiver_rank) {
    if ((int)mdmp_decl_req_counter + MDMP_MAX_REQUESTS >= MDMP_DECL_BATCH_TOKEN_BASE) {
      fprintf(stderr, "[MDMP FATAL] Declarative logical ID space exhausted.\n");
      mdmp_abort(1);
    }    
    uint32_t id = mdmp_decl_req_counter++;
    recv_queue.push_back({buffer, count, type, bytes, src_rank, tag, (int)id}); 
    return (int)id + MDMP_MAX_REQUESTS;
  }
  return MDMP_PROCESS_NOT_INVOLVED;
}

int mdmp_register_reduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes, int root_rank, int op) {
  uint32_t id = mdmp_decl_req_counter++;
  if ((int)mdmp_decl_req_counter + MDMP_MAX_REQUESTS >= MDMP_DECL_BATCH_TOKEN_BASE) {
    fprintf(stderr, "[MDMP FATAL] Declarative logical ID space exhausted.\n");
    mdmp_abort(1);
  }  
  reduce_queue.push_back({sendbuf, recvbuf, count, type, bytes, root_rank, op, (int)id});
  return (int)id + MDMP_MAX_REQUESTS;
}

int mdmp_register_gather(void* sendbuf, size_t sendcount, void* recvbuf, int type, size_t bytes, int root_rank) {
  uint32_t id = mdmp_decl_req_counter++;
  if ((int)mdmp_decl_req_counter + MDMP_MAX_REQUESTS >= MDMP_DECL_BATCH_TOKEN_BASE) {
    fprintf(stderr, "[MDMP FATAL] Declarative logical ID space exhausted.\n");
    mdmp_abort(1);
  }  
  gather_queue.push_back({sendbuf, sendcount, recvbuf, type, bytes, root_rank, (int)id});
  return (int)id + MDMP_MAX_REQUESTS;
}

int mdmp_register_allreduce(void* sendbuf, void* recvbuf, size_t count, int type,  size_t bytes, int op) {
  uint32_t id = mdmp_decl_req_counter++;
  if ((int)mdmp_decl_req_counter + MDMP_MAX_REQUESTS >= MDMP_DECL_BATCH_TOKEN_BASE) {
    fprintf(stderr, "[MDMP FATAL] Declarative logical ID space exhausted.\n");
    mdmp_abort(1);
  }
  allreduce_queue.push_back({sendbuf, recvbuf, count, type, bytes, op, (int)id});
  return (int)id + MDMP_MAX_REQUESTS;
}

int mdmp_register_allgather(void* sendbuf, size_t count, void* recvbuf, int type, size_t bytes) {
  uint32_t id = mdmp_decl_req_counter++;
  if ((int)mdmp_decl_req_counter + MDMP_MAX_REQUESTS >= MDMP_DECL_BATCH_TOKEN_BASE) {
    fprintf(stderr, "[MDMP FATAL] Declarative logical ID space exhausted.\n");
    mdmp_abort(1);
  }
  allgather_queue.push_back({sendbuf, count, recvbuf, type, bytes, (int)id});
  return (int)id + MDMP_MAX_REQUESTS;
}

int mdmp_register_bcast(void* buffer, size_t count, int type, size_t bytes, int root_rank) {
  uint32_t id = mdmp_decl_req_counter++;
  if ((int)mdmp_decl_req_counter + MDMP_MAX_REQUESTS >= MDMP_DECL_BATCH_TOKEN_BASE) {
    fprintf(stderr, "[MDMP FATAL] Declarative logical ID space exhausted.\n");
    mdmp_abort(1);
  }
  bcast_queue.push_back({buffer, count, type, bytes, root_rank, (int)id});
  return (int)id + MDMP_MAX_REQUESTS;
}

int mdmp_commit() {
  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (mdmp_runtime_active)
    lock.lock();

  if (mdmp_decl_batch_counter >= (uint32_t)(INT_MAX - MDMP_DECL_BATCH_TOKEN_BASE)) {
    fprintf(stderr, "[MDMP FATAL] Declarative batch token space exhausted.\n");
    mdmp_abort(1);
  }

  DeclarativeBatch Batch;
  Batch.Serial = mdmp_decl_batch_counter++;

  static std::vector<int> blens_buffer;
  static std::vector<MPI_Aint> disps_buffer;

  auto MapLogicalReq = [&](int logicalReqID, int mpi_idx) {
    decl_to_req_target[logicalReqID] = {Batch.Serial, mpi_idx};
    Batch.LogicalReqIDs.push_back(logicalReqID);
  };

  auto ProcessQueue = [&](std::vector<RegisteredMsg>& queue, bool isSend) {
    if (queue.empty()) return;

    std::stable_sort(queue.begin(), queue.end(),
                     [](const RegisteredMsg& a, const RegisteredMsg& b) {
                       if (a.rank != b.rank) return a.rank < b.rank;
                       return a.tag < b.tag;
                     });

    size_t i = 0;
    while (i < queue.size()) {
      int peer = queue[i].rank;
      int current_tag = queue[i].tag;
      size_t j = i + 1;

      while (j < queue.size() &&
             queue[j].rank == peer &&
             queue[j].tag == current_tag) {
        j++;
      }

      size_t count = j - i;
      int mpi_idx = (int)Batch.Requests.size();
      Batch.Requests.push_back(MPI_REQUEST_NULL);

      if (count == 1) {
        int actual_count = (queue[i].type == 4) ? (int)queue[i].bytes : (int)queue[i].count;

        if (isSend) {
          mdmp_check_mpi(
              MPI_Isend(queue[i].buffer, actual_count, get_mpi_type(queue[i].type),
                        peer, current_tag, mdmp_comm, &Batch.Requests[mpi_idx]),
              "MPI_Isend(declarative)");
        } else {
          mdmp_check_mpi(
              MPI_Irecv(queue[i].buffer, actual_count, get_mpi_type(queue[i].type),
                        peer, current_tag, mdmp_comm, &Batch.Requests[mpi_idx]),
              "MPI_Irecv(declarative)");
        }

        MapLogicalReq(queue[i].req_id, mpi_idx);
      } else {
        blens_buffer.resize(count);
        disps_buffer.resize(count);

        for (size_t k = 0; k < count; ++k) {
          int element_bytes = (int)queue[i + k].bytes;
          if (element_bytes == 0) {
            int mpi_type_size = 0;
            mdmp_check_mpi(
                MPI_Type_size(get_mpi_type(queue[i + k].type), &mpi_type_size),
                "MPI_Type_size");
            element_bytes = (int)queue[i + k].count * mpi_type_size;
          }

          blens_buffer[k] = element_bytes;
          mdmp_check_mpi(
              MPI_Get_address(queue[i + k].buffer, &disps_buffer[k]),
              "MPI_Get_address");
        }

        MPI_Datatype ntype = MPI_DATATYPE_NULL;
        mdmp_check_mpi(
            MPI_Type_create_hindexed((int)count, blens_buffer.data(), disps_buffer.data(),
                                     MPI_BYTE, &ntype),
            "MPI_Type_create_hindexed");
        mdmp_check_mpi(
            MPI_Type_commit(&ntype),
            "MPI_Type_commit");

        Batch.TypesToFree.push_back(ntype);

        if (isSend) {
          mdmp_check_mpi(
              MPI_Isend(MPI_BOTTOM, 1, ntype, peer, current_tag, mdmp_comm,
                        &Batch.Requests[mpi_idx]),
              "MPI_Isend(declarative hindexed)");
        } else {
          mdmp_check_mpi(
              MPI_Irecv(MPI_BOTTOM, 1, ntype, peer, current_tag, mdmp_comm,
                        &Batch.Requests[mpi_idx]),
              "MPI_Irecv(declarative hindexed)");
        }

        for (size_t k = 0; k < count; ++k) {
          MapLogicalReq(queue[i + k].req_id, mpi_idx);
        }
      }

      i = j;
    }
  };

  ProcessQueue(recv_queue, false);
  ProcessQueue(send_queue, true);

  send_queue.clear();
  recv_queue.clear();

  for (auto &r : reduce_queue) {
    int mpi_idx = (int)Batch.Requests.size();
    Batch.Requests.push_back(MPI_REQUEST_NULL);

    const void* final_sb =
        (r.sendbuf == r.recvbuf && global_my_rank == r.root) ? MPI_IN_PLACE : r.sendbuf;
    int actual_count = (r.type == 4) ? (int)r.bytes : (int)r.count;

    mdmp_check_mpi(
        MPI_Ireduce(final_sb, r.recvbuf, actual_count, get_mpi_type(r.type),
                    get_mpi_op(r.op), r.root, mdmp_comm, &Batch.Requests[mpi_idx]),
        "MPI_Ireduce(declarative)");

    MapLogicalReq(r.req_id, mpi_idx);
  }

  for (auto &g : gather_queue) {
    int mpi_idx = (int)Batch.Requests.size();
    Batch.Requests.push_back(MPI_REQUEST_NULL);

    const void* final_sb =
        (g.sendbuf == g.recvbuf && global_my_rank == g.root) ? MPI_IN_PLACE : g.sendbuf;
    int actual_count = (g.type == 4) ? (int)g.bytes : (int)g.sendcount;

    mdmp_check_mpi(
        MPI_Igather(final_sb, actual_count, get_mpi_type(g.type),
                    g.recvbuf, actual_count, get_mpi_type(g.type),
                    g.root, mdmp_comm, &Batch.Requests[mpi_idx]),
        "MPI_Igather(declarative)");

    MapLogicalReq(g.req_id, mpi_idx);
  }

  for (auto &ar : allreduce_queue) {
    int mpi_idx = (int)Batch.Requests.size();
    Batch.Requests.push_back(MPI_REQUEST_NULL);

    const void* final_sb = (ar.sendbuf == ar.recvbuf) ? MPI_IN_PLACE : ar.sendbuf;
    int actual_count = (ar.type == 4) ? (int)ar.bytes : (int)ar.count;

    mdmp_check_mpi(
        MPI_Iallreduce(final_sb, ar.recvbuf, actual_count, get_mpi_type(ar.type),
                       get_mpi_op(ar.op), mdmp_comm, &Batch.Requests[mpi_idx]),
        "MPI_Iallreduce(declarative)");

    MapLogicalReq(ar.req_id, mpi_idx);
  }

  for (auto &ag : allgather_queue) {
    int mpi_idx = (int)Batch.Requests.size();
    Batch.Requests.push_back(MPI_REQUEST_NULL);

    const void* final_sb = (ag.sendbuf == ag.recvbuf) ? MPI_IN_PLACE : ag.sendbuf;
    int actual_count = (ag.type == 4) ? (int)ag.bytes : (int)ag.count;

    mdmp_check_mpi(
        MPI_Iallgather(final_sb, actual_count, get_mpi_type(ag.type),
                       ag.recvbuf, actual_count, get_mpi_type(ag.type),
                       mdmp_comm, &Batch.Requests[mpi_idx]),
        "MPI_Iallgather(declarative)");

    MapLogicalReq(ag.req_id, mpi_idx);
  }

  for (auto &b : bcast_queue) {
    int mpi_idx = (int)Batch.Requests.size();
    Batch.Requests.push_back(MPI_REQUEST_NULL);

    int actual_count = (b.type == 4) ? (int)b.bytes : (int)b.count;

    mdmp_check_mpi(
        MPI_Ibcast(b.buffer, actual_count, get_mpi_type(b.type),
                   b.root, mdmp_comm, &Batch.Requests[mpi_idx]),
        "MPI_Ibcast(declarative)");

    MapLogicalReq(b.req_id, mpi_idx);
  }

  reduce_queue.clear();
  gather_queue.clear();
  allreduce_queue.clear();
  allgather_queue.clear();
  bcast_queue.clear();

  // If this rank ended up with no actual requests in this commit, return a no-op token.
  if (Batch.Requests.empty()) {
    return MDMP_PROCESS_NOT_INVOLVED;
  }

  uint32_t Serial = Batch.Serial;
  auto Inserted = mdmp_active_decl_batches.emplace(Serial, std::move(Batch));
  if (!Inserted.second) {
    fprintf(stderr, "[MDMP FATAL] Declarative batch serial collision: %u\n", Serial);
    mdmp_abort(1);
  }

  return mdmp_make_decl_batch_token(Serial);
}
