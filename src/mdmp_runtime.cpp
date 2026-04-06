#include "mdmp_runtime.h"

static int global_my_rank = -1;
static int global_size = -1;
static int mdmp_debug_enabled = 0;

MPI_Comm mdmp_comm; 
std::atomic<bool> mdmp_runtime_active{false};
std::thread mdmp_progress_thread;
std::mutex mdmp_mpi_mutex;
bool mdmp_owns_mpi = false;

// Profiling variables
static int mdmp_profile_enabled = 0;

static uint64_t mdmp_prof_maybe_progress_calls = 0;
static uint64_t mdmp_prof_maybe_progress_skipped_thread = 0;
static uint64_t mdmp_prof_maybe_progress_skipped_idle = 0;

static uint64_t mdmp_prof_progress_calls = 0;
static uint64_t mdmp_prof_progress_tested_imperative = 0;
static uint64_t mdmp_prof_progress_tested_declarative = 0;
static uint64_t mdmp_prof_progress_time_us = 0;

static uint64_t mdmp_prof_wait_imp_calls = 0;
static uint64_t mdmp_prof_wait_imp_fast = 0;
static uint64_t mdmp_prof_wait_imp_blocking = 0;
static uint64_t mdmp_prof_wait_imp_block_time_us = 0;

static uint64_t mdmp_prof_wait_decl_logical_calls = 0;
static uint64_t mdmp_prof_wait_decl_logical_fast = 0;
static uint64_t mdmp_prof_wait_decl_logical_blocking = 0;
static uint64_t mdmp_prof_wait_decl_logical_block_time_us = 0;

static uint64_t mdmp_prof_wait_decl_batch_calls = 0;
static uint64_t mdmp_prof_wait_decl_batch_fast = 0;
static uint64_t mdmp_prof_wait_decl_batch_blocking = 0;
static uint64_t mdmp_prof_wait_decl_batch_block_time_us = 0;

static uint64_t mdmp_prof_send_calls = 0;
static uint64_t mdmp_prof_send_bytes = 0;
static uint64_t mdmp_prof_send_time_us = 0;

static uint64_t mdmp_prof_recv_calls = 0;
static uint64_t mdmp_prof_recv_bytes = 0;
static uint64_t mdmp_prof_recv_time_us = 0;

static uint64_t mdmp_prof_wait_many_calls = 0;
static uint64_t mdmp_prof_wait_many_ids_total = 0;
static uint64_t mdmp_prof_wait_many_time_us = 0;

static uint64_t mdmp_prof_wait_many_fast_calls = 0;
static uint64_t mdmp_prof_wait_many_tiny_seq_calls = 0;
static uint64_t mdmp_prof_wait_many_fallback_calls = 0;

static uint64_t mdmp_prof_imp_wait_scan_calls = 0;
static uint64_t mdmp_prof_imp_wait_scan_slots = 0;
static uint64_t mdmp_prof_imp_wait_active_found = 0;
static uint64_t mdmp_prof_imp_wait_scan_time_us = 0;

struct MDMPProgressSiteStats {
  uint64_t Hits = 0;
  uint64_t IdleSkips = 0;
  uint64_t ActiveHits = 0;
  uint64_t ProgressCalls = 0;
};

static std::vector<MDMPProgressSiteStats> mdmp_prof_progress_sites;

// ==============================================================================
// ZERO-ALLOCATION STATIC QUEUES (DECLARATIVE API)
// ==============================================================================
static constexpr int MDMP_DECL_BATCH_TOKEN_BASE = (1 << 30);
#define MDMP_STATIC_Q_SIZE 2048

struct RegisteredMsg { void* buffer; size_t count; int type; size_t bytes; int rank; int tag; int req_id; };
static RegisteredMsg send_queue[MDMP_STATIC_Q_SIZE];
static int send_q_count = 0;

static RegisteredMsg recv_queue[MDMP_STATIC_Q_SIZE];
static int recv_q_count = 0;

struct RegisteredReduce { void* sendbuf; void* recvbuf; size_t count; int type; size_t bytes; int root; int op; int req_id; };
static RegisteredReduce reduce_queue[MDMP_STATIC_Q_SIZE];
static int reduce_q_count = 0;

struct RegisteredGather { void* sendbuf; size_t sendcount; void* recvbuf; int type; size_t bytes; int root; int req_id; };
static RegisteredGather gather_queue[MDMP_STATIC_Q_SIZE];
static int gather_q_count = 0;

struct RegisteredAllreduce { void* sendbuf; void* recvbuf; size_t count; int type; size_t bytes; int op; int req_id; };
static RegisteredAllreduce allreduce_queue[MDMP_STATIC_Q_SIZE];
static int allreduce_q_count = 0;

struct RegisteredAllgather { void* sendbuf; size_t count; void* recvbuf; int type; size_t bytes; int req_id; };
static RegisteredAllgather allgather_queue[MDMP_STATIC_Q_SIZE];
static int allgather_q_count = 0;

struct RegisteredBcast { void* buffer; size_t count; int type; size_t bytes; int root; int req_id; };
static RegisteredBcast bcast_queue[MDMP_STATIC_Q_SIZE];
static int bcast_q_count = 0;

struct DeclarativeBatch {
  uint32_t Serial = 0;
  MPI_Request Requests[MDMP_STATIC_Q_SIZE * 4];
  int RequestCount = 0;
  bool Active = false;
};

// Array of pre-allocated batches to round-robin through
static DeclarativeBatch mdmp_batch_pool[16]; 

struct DeclWaitTarget {
  uint32_t BatchSerial;
  int MPIIndex;
};
// Flat map for looking up specific logical IDs
static DeclWaitTarget decl_to_req_target[65536];

// MPI request tracking
MPI_Request mdmp_request_pool[MDMP_MAX_REQUESTS];

// Free-list of available imperative request slots
int mdmp_imper_free_slots[MDMP_MAX_REQUESTS];
int mdmp_imper_free_top = 0;

// Dense active list of currently in-flight imperative slots
int mdmp_imper_active_slots[MDMP_MAX_REQUESTS];
int mdmp_imper_active_pos[MDMP_MAX_REQUESTS];
int mdmp_imper_active_count = 0;

uint32_t mdmp_decl_req_counter = 0;    
uint32_t mdmp_decl_batch_counter = 0;  

std::atomic<int> mdmp_active_request_count{0};


static inline void mdmp_profile_ensure_site(int site_id) {
  if (site_id < 0)
    return;

  size_t need = static_cast<size_t>(site_id) + 1;
  if (mdmp_prof_progress_sites.size() < need)
    mdmp_prof_progress_sites.resize(need);
}

// Fast pipeline bypass for atomic increments
static inline void mdmp_note_requests_started(int n) {
  if (n > 0)
    mdmp_active_request_count.fetch_add(n, std::memory_order_relaxed);
}

static inline void mdmp_note_requests_completed(int n) {
  if (n > 0)
    mdmp_active_request_count.fetch_sub(n, std::memory_order_relaxed);
}

uint64_t mdmp_now_us() {
  return (uint64_t)(MPI_Wtime() * 1.0e6);
}

void mdmp_profile_reset() {
  mdmp_prof_maybe_progress_calls = 0;
  mdmp_prof_maybe_progress_skipped_thread = 0;
  mdmp_prof_maybe_progress_skipped_idle = 0;
  
  mdmp_prof_progress_calls = 0;
  mdmp_prof_progress_tested_imperative = 0;
  mdmp_prof_progress_tested_declarative = 0;
  
  mdmp_prof_progress_time_us = 0;

  mdmp_prof_send_calls = 0;
  mdmp_prof_send_bytes = 0;
  mdmp_prof_send_time_us = 0;

  mdmp_prof_recv_calls = 0;
  mdmp_prof_recv_bytes = 0;
  mdmp_prof_recv_time_us = 0;
  
  mdmp_prof_wait_imp_calls = 0;
  mdmp_prof_wait_imp_fast = 0;
  mdmp_prof_wait_imp_blocking = 0;
  mdmp_prof_wait_imp_block_time_us = 0;
  
  mdmp_prof_wait_decl_logical_calls = 0;
  mdmp_prof_wait_decl_logical_fast = 0;
  mdmp_prof_wait_decl_logical_blocking = 0;
  mdmp_prof_wait_decl_logical_block_time_us = 0;
  
  mdmp_prof_wait_decl_batch_calls = 0;
  mdmp_prof_wait_decl_batch_fast = 0;
  mdmp_prof_wait_decl_batch_blocking = 0;
  mdmp_prof_wait_decl_batch_block_time_us = 0;

  mdmp_prof_wait_many_calls = 0;
  mdmp_prof_wait_many_ids_total = 0;
  mdmp_prof_wait_many_time_us = 0;

  mdmp_prof_wait_many_fast_calls = 0;
  mdmp_prof_wait_many_tiny_seq_calls = 0;
  mdmp_prof_wait_many_fallback_calls = 0;

  mdmp_prof_progress_sites.clear();

}

void mdmp_profile_report() {
  if (!mdmp_profile_enabled) return;

  uint64_t local_basic[15] = {
    mdmp_prof_maybe_progress_calls,
    mdmp_prof_maybe_progress_skipped_thread,
    mdmp_prof_maybe_progress_skipped_idle,
    mdmp_prof_progress_calls,
    mdmp_prof_progress_tested_imperative,
    mdmp_prof_progress_tested_declarative,
    mdmp_prof_progress_time_us,
    mdmp_prof_wait_imp_calls,
    mdmp_prof_wait_imp_fast,
    mdmp_prof_wait_imp_blocking,
    mdmp_prof_wait_imp_block_time_us,
    mdmp_prof_wait_decl_logical_calls,
    mdmp_prof_wait_decl_logical_fast,
    mdmp_prof_wait_decl_logical_blocking,
    mdmp_prof_wait_decl_logical_block_time_us
  };

  uint64_t global_basic[15] = {0};
  mdmp_check_mpi(MPI_Reduce(local_basic, global_basic, 15, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, mdmp_comm), "MPI_Reduce");

  uint64_t local_batch[4] = {
    mdmp_prof_wait_decl_batch_calls,
    mdmp_prof_wait_decl_batch_fast,
    mdmp_prof_wait_decl_batch_blocking,
    mdmp_prof_wait_decl_batch_block_time_us
  };

  uint64_t global_batch[4] = {0};
  mdmp_check_mpi(MPI_Reduce(local_batch, global_batch, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, mdmp_comm), "MPI_Reduce");

  uint64_t local_wait_many[6] = {
    mdmp_prof_wait_many_calls,
    mdmp_prof_wait_many_ids_total,
    mdmp_prof_wait_many_time_us,
    mdmp_prof_wait_many_fast_calls,
    mdmp_prof_wait_many_tiny_seq_calls,
    mdmp_prof_wait_many_fallback_calls
  };

  uint64_t global_wait_many[6] = {0};

  mdmp_check_mpi(
		 MPI_Reduce(local_wait_many, global_wait_many, 6, MPI_UNSIGNED_LONG_LONG,
			    MPI_SUM, 0, mdmp_comm),
		 "MPI_Reduce(mdmp_profile wait_many)");

  uint64_t local_comm[6] = {
    mdmp_prof_send_calls,
    mdmp_prof_send_bytes,
    mdmp_prof_send_time_us,
    mdmp_prof_recv_calls,
    mdmp_prof_recv_bytes,
    mdmp_prof_recv_time_us
  };

  uint64_t global_comm[6] = {0};

  mdmp_check_mpi(
		 MPI_Reduce(local_comm, global_comm, 6, MPI_UNSIGNED_LONG_LONG,
			    MPI_SUM, 0, mdmp_comm),
		 "MPI_Reduce(mdmp_profile comm)");

  

  if (global_my_rank == 0) {
    auto pct = [](uint64_t num, uint64_t den) -> double {
      return (den == 0) ? 0.0 : (100.0 * (double)num / (double)den);
    };
    printf("\n[MDMP PROFILE] Global summary across %d ranks\n", global_size);
    printf("  maybe_progress calls                 : %llu\n", (unsigned long long)global_basic[0]);
    printf("  imperative waits                     : %llu\n", (unsigned long long)global_basic[7]);
    printf("    already complete                   : %llu (%.1f%%)\n", (unsigned long long)global_basic[8], pct(global_basic[8], global_basic[7]));
    printf("    blocking                           : %llu (%.1f%%)\n", (unsigned long long)global_basic[9], pct(global_basic[9], global_basic[7]));
    printf("    total block time (s)               : %.6f\n", (double)global_basic[10] / 1.0e6);
    printf("  declarative batch waits              : %llu\n", (unsigned long long)global_batch[0]);
    printf("    already complete                   : %llu (%.1f%%)\n", (unsigned long long)global_batch[1], pct(global_batch[1], global_batch[0]));
    printf("    blocking                           : %llu (%.1f%%)\n", (unsigned long long)global_batch[2], pct(global_batch[2], global_batch[0]));
    printf("    total block time (s)               : %.6f\n", (double)global_batch[3] / 1.0e6);
    printf("\n");
    printf("  wait_many calls                      : %llu\n",
           (unsigned long long)global_wait_many[0]);
    printf("    total ids passed                   : %llu\n",
           (unsigned long long)global_wait_many[1]);
    printf("    total time (s)                     : %.6f\n",
           (double)global_wait_many[2] / 1.0e6);
    printf("    micro fast-path calls              : %llu\n",
           (unsigned long long)global_wait_many[3]);
    printf("    tiny sequential calls              : %llu\n",
           (unsigned long long)global_wait_many[4]);
    printf("    large fallback calls               : %llu\n",
           (unsigned long long)global_wait_many[5]);
    printf("  send calls                            : %llu\n",
           (unsigned long long)global_comm[0]);
    printf("    total bytes                         : %llu\n",
           (unsigned long long)global_comm[1]);
    printf("    total time (s)                      : %.6f\n",
           (double)global_comm[2] / 1.0e6);

    printf("  recv calls                            : %llu\n",
           (unsigned long long)global_comm[3]);
    printf("    total bytes                         : %llu\n",
           (unsigned long long)global_comm[4]);
    printf("    total time (s)                      : %.6f\n",
           (double)global_comm[5] / 1.0e6);

  }

  int local_num_sites = static_cast<int>(mdmp_prof_progress_sites.size());
  int global_num_sites = 0;

  mdmp_check_mpi(
		 MPI_Allreduce(&local_num_sites, &global_num_sites, 1, MPI_INT,
			       MPI_MAX, mdmp_comm),
		 "MPI_Allreduce(mdmp_profile site count)");

  if (global_num_sites > 0) {
    std::vector<uint64_t> local_site_data(4 * global_num_sites, 0);
    for (int i = 0; i < local_num_sites; ++i) {
      local_site_data[4 * i + 0] = mdmp_prof_progress_sites[i].Hits;
      local_site_data[4 * i + 1] = mdmp_prof_progress_sites[i].IdleSkips;
      local_site_data[4 * i + 2] = mdmp_prof_progress_sites[i].ActiveHits;
      local_site_data[4 * i + 3] = mdmp_prof_progress_sites[i].ProgressCalls;
    }

    std::vector<uint64_t> global_site_data;
    if (global_my_rank == 0) {
      global_site_data.resize(4 * global_num_sites, 0);
    }

    mdmp_check_mpi(
		   MPI_Reduce(local_site_data.data(),
			      global_my_rank == 0 ? global_site_data.data() : nullptr,
			      4 * global_num_sites,
			      MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, mdmp_comm),
		   "MPI_Reduce(mdmp_profile site data)");

    if (global_my_rank == 0) {
      struct SiteRow {
        int SiteID;
        uint64_t Hits;
        uint64_t IdleSkips;
        uint64_t ActiveHits;
        uint64_t ProgressCalls;
      };

      std::vector<SiteRow> rows;
      rows.reserve(global_num_sites);

      for (int i = 0; i < global_num_sites; ++i) {
        uint64_t hits = global_site_data[4 * i + 0];
        uint64_t idle = global_site_data[4 * i + 1];
        uint64_t active = global_site_data[4 * i + 2];
        uint64_t prog = global_site_data[4 * i + 3];

        if (hits == 0)
          continue;

        rows.push_back({i, hits, idle, active, prog});
      }

      std::stable_sort(rows.begin(), rows.end(),
                       [](const SiteRow &A, const SiteRow &B) {
                         if (A.ActiveHits != B.ActiveHits)
                           return A.ActiveHits > B.ActiveHits;
                         if (A.ProgressCalls != B.ProgressCalls)
                           return A.ProgressCalls > B.ProgressCalls;
                         return A.Hits > B.Hits;
                       });

      printf("[MDMP PROFILE] Top progress sites (by active hits)\n");

      int to_print = std::min<int>(10, rows.size());
      for (int i = 0; i < to_print; ++i) {
        const SiteRow &R = rows[i];
        printf("  site %d: hits=%llu idle=%llu active=%llu progress_calls=%llu\n",
               R.SiteID,
               (unsigned long long)R.Hits,
               (unsigned long long)R.IdleSkips,
               (unsigned long long)R.ActiveHits,
               (unsigned long long)R.ProgressCalls);
      }

      printf("\n");
    }
  }

}

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

inline uint64_t mdmp_actual_bytes(size_t count, int type, size_t bytes) {
  switch (type) {
  case 0: return (uint64_t)count * 4; // int
  case 1: return (uint64_t)count * 8; // double
  case 2: return (uint64_t)count * 4; // float
  case 3: return (uint64_t)count;     // char
  case 4: return (uint64_t)bytes;     // raw bytes
  default:
    return (bytes != 0) ? (uint64_t)bytes : (uint64_t)count;
  }
}

int mdmp_alloc_imperative_slot_unlocked() {
  if (mdmp_imper_free_top == 0) {
    fprintf(stderr,
            "[MDMP FATAL] Request pool overflow! Exceeded %d concurrent imperative requests.\n",
            MDMP_MAX_REQUESTS);
    mdmp_abort(1);
  }

  int id = mdmp_imper_free_slots[--mdmp_imper_free_top];

  mdmp_imper_active_pos[id] = mdmp_imper_active_count;
  mdmp_imper_active_slots[mdmp_imper_active_count++] = id;

  return id;
}

static inline void mdmp_release_imperative_slot_unlocked(int id) {
  if (id < 0 || id >= MDMP_MAX_REQUESTS)
    return;

  int pos = mdmp_imper_active_pos[id];
  if (pos < 0)
    return;

  int last_slot = mdmp_imper_active_slots[mdmp_imper_active_count - 1];
  mdmp_imper_active_slots[pos] = last_slot;
  mdmp_imper_active_pos[last_slot] = pos;

  mdmp_imper_active_count--;
  mdmp_imper_active_pos[id] = -1;
  mdmp_imper_free_slots[mdmp_imper_free_top++] = id;
}

inline bool mdmp_is_decl_batch_token(int token) { return token >= MDMP_DECL_BATCH_TOKEN_BASE; }
inline int mdmp_make_decl_batch_token(uint32_t serial) { return MDMP_DECL_BATCH_TOKEN_BASE + (int)serial; }
inline uint32_t mdmp_decl_batch_serial_from_token(int token) { return (uint32_t)(token - MDMP_DECL_BATCH_TOKEN_BASE); }

void mdmp_wait_declarative_batch_unlocked(uint32_t serial) {
  DeclarativeBatch& B = mdmp_batch_pool[serial % 16];
  if (B.Serial == serial && B.Active) {
    
    if (mdmp_profile_enabled) {
      mdmp_prof_wait_decl_batch_calls++;
      int all_done = 0;
      MPI_Testall(B.RequestCount, B.Requests, &all_done, MPI_STATUSES_IGNORE);
      if (all_done) {
        mdmp_prof_wait_decl_batch_fast++;
      } else {
        mdmp_prof_wait_decl_batch_blocking++;
        uint64_t t0 = mdmp_now_us();
        
        bool use_locks = mdmp_runtime_active.load(std::memory_order_relaxed);
        if (use_locks) mdmp_mpi_mutex.unlock();
        MPI_Waitall(B.RequestCount, B.Requests, MPI_STATUSES_IGNORE);
        if (use_locks) mdmp_mpi_mutex.lock();
        
        mdmp_prof_wait_decl_batch_block_time_us += (mdmp_now_us() - t0);
      }
    } else {
      bool use_locks = mdmp_runtime_active.load(std::memory_order_relaxed);
      if (use_locks) mdmp_mpi_mutex.unlock();
      mdmp_check_mpi(MPI_Waitall(B.RequestCount, B.Requests, MPI_STATUSES_IGNORE), "MPI_Waitall(declarative batch)");
      if (use_locks) mdmp_mpi_mutex.lock();
    }

    mdmp_note_requests_completed(B.RequestCount);
    B.Active = false;
  }
}

void mdmp_wait_all_declarative_batches_unlocked() {
  for (int i = 0; i < 16; i++) {
    if (mdmp_batch_pool[i].Active) {
      mdmp_wait_declarative_batch_unlocked(mdmp_batch_pool[i].Serial);
    }
  }
}

bool mdmp_any_declarative_batches_active() {
  for (int i = 0; i < 16; i++) {
    if (mdmp_batch_pool[i].Active) return true;
  }
  return false;
}

void mdmp_check_mpi(int rc, const char *where) {
  if (rc == MPI_SUCCESS) return;
  char errstr[MPI_MAX_ERROR_STRING];
  int len = 0;
  MPI_Error_string(rc, errstr, &len);
  fprintf(stderr, "[MDMP FATAL] Rank %d: %s failed with MPI error: %.*s\n", global_my_rank, where, len, errstr);
  mdmp_abort(rc);
}

bool mdmp_has_active_requests() {
  return mdmp_active_request_count.load(std::memory_order_relaxed) > 0;
}

void mdmp_progress() {
  int flag;
  uint64_t tested_imp = 0;
  
  for (int idx = 0; idx < mdmp_imper_active_count; ) {
    int slot = mdmp_imper_active_slots[idx];
    int flag_local = 0;

    MPI_Test(&mdmp_request_pool[slot], &flag_local, MPI_STATUS_IGNORE);
    tested_imp++;

    if (flag_local && mdmp_request_pool[slot] == MPI_REQUEST_NULL) {
      mdmp_note_requests_completed(1);
      mdmp_release_imperative_slot_unlocked(slot);
      // do not increment idx: active list has been compacted
    } else {
      ++idx;
    }
  }

  for (int i = 0; i < 16; i++) {
    if (mdmp_batch_pool[i].Active) {
      for (int j = 0; j < mdmp_batch_pool[i].RequestCount; j++) {
        if (mdmp_batch_pool[i].Requests[j] != MPI_REQUEST_NULL) {
          MPI_Test(&mdmp_batch_pool[i].Requests[j], &flag, MPI_STATUS_IGNORE);
          if (flag && mdmp_batch_pool[i].Requests[j] == MPI_REQUEST_NULL) {
            mdmp_note_requests_completed(1);
          }
        }
      }
      bool all_done = true;
      for (int j = 0; j < mdmp_batch_pool[i].RequestCount; j++) {
        if (mdmp_batch_pool[i].Requests[j] != MPI_REQUEST_NULL) { all_done = false; break; }
      }
      if (all_done) mdmp_batch_pool[i].Active = false;
    }
  }
}

void mdmp_progress_loop() {
  while (mdmp_runtime_active) {
    {
      std::lock_guard<std::mutex> lock(mdmp_mpi_mutex);
      mdmp_progress();
    }
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}

void mdmp_maybe_progress_site(int site_id) {
  if (mdmp_profile_enabled) {
    mdmp_prof_maybe_progress_calls++;
    mdmp_profile_ensure_site(site_id);
    if (site_id >= 0) {
      mdmp_prof_progress_sites[site_id].Hits++;
    }
  }

  if (mdmp_runtime_active) {
    if (mdmp_profile_enabled)
      mdmp_prof_maybe_progress_skipped_thread++;
    return;
  }

  if (mdmp_active_request_count.load(std::memory_order_relaxed) == 0) {
    if (mdmp_profile_enabled) {
      mdmp_prof_maybe_progress_skipped_idle++;
      if (site_id >= 0) {
        mdmp_prof_progress_sites[site_id].IdleSkips++;
      }
    }
    return;
  }

  if (mdmp_profile_enabled && site_id >= 0) {
    mdmp_prof_progress_sites[site_id].ActiveHits++;
  }

  mdmp_progress();

  if (mdmp_profile_enabled && site_id >= 0) {
    mdmp_prof_progress_sites[site_id].ProgressCalls++;
  }
}

void mdmp_maybe_progress() {
  mdmp_maybe_progress_site(-1);
}

void mdmp_wait_unlocked(int req_id) {
  if (req_id == MDMP_PROCESS_NOT_INVOLVED) return;
  
  // 1. FAST PATH: IMPLICIT WAIT BATCHING
  if (req_id >= 0 && req_id < MDMP_MAX_REQUESTS) {
    if (mdmp_request_pool[req_id] != MPI_REQUEST_NULL) {

      uint64_t scan_t0 = 0;
      if (mdmp_profile_enabled) {
	mdmp_prof_imp_wait_scan_calls++;
	scan_t0 = mdmp_now_us();
      }
            
      MPI_Request stack_reqs[32];
      int active_slots[32];
      int active_count = 0;

      // Use the dense active set, not a full pool scan.
      for (int i = 0; i < mdmp_imper_active_count; ++i) {
        int slot = mdmp_imper_active_slots[i];
        if (mdmp_request_pool[slot] != MPI_REQUEST_NULL) {
          stack_reqs[active_count] = mdmp_request_pool[slot];
          active_slots[active_count] = slot;
          active_count++;
          if (active_count == 32) break;
        }
      }

      if (active_count == 1) {
        int slot = active_slots[0];
        mdmp_check_mpi(
		       MPI_Wait(&mdmp_request_pool[slot], MPI_STATUS_IGNORE),
		       "MPI_Wait(imperative)");
        mdmp_request_pool[slot] = MPI_REQUEST_NULL;
        mdmp_note_requests_completed(1);
        mdmp_release_imperative_slot_unlocked(slot);
      } else if (active_count > 1) {
        bool use_locks = mdmp_runtime_active.load(std::memory_order_relaxed);
        if (use_locks) mdmp_mpi_mutex.unlock();

        mdmp_check_mpi(
		       MPI_Waitall(active_count, stack_reqs, MPI_STATUSES_IGNORE),
		       "MPI_Waitall(implicit batch)");

        if (use_locks) mdmp_mpi_mutex.lock();

        for (int i = 0; i < active_count; ++i) {
          int slot = active_slots[i];
          mdmp_request_pool[slot] = MPI_REQUEST_NULL;
          mdmp_release_imperative_slot_unlocked(slot);
        }
        mdmp_note_requests_completed(active_count);
      }
      
      if (mdmp_profile_enabled) {
	mdmp_prof_imp_wait_scan_slots += mdmp_imper_active_count; // if using dense list
	mdmp_prof_imp_wait_active_found += active_count;
	mdmp_prof_imp_wait_scan_time_us += (mdmp_now_us() - scan_t0);
      }
      
    }
    return;
  }

  
  // 2. Whole declarative batch token
  if (mdmp_is_decl_batch_token(req_id)) {
    uint32_t batchSerial = mdmp_decl_batch_serial_from_token(req_id);
    mdmp_wait_declarative_batch_unlocked(batchSerial);
    return;
  }

  // 3. Specific declarative logical request
  if (req_id >= MDMP_MAX_REQUESTS && req_id != MDMP_DECLARATIVE_WAIT) {
    int logicalReqID = req_id - MDMP_MAX_REQUESTS;
    DeclWaitTarget& Target = decl_to_req_target[logicalReqID % 65536];
    uint32_t batchSerial = Target.BatchSerial;
    DeclarativeBatch &B = mdmp_batch_pool[batchSerial % 16];
    
    if (B.Serial == batchSerial && B.Active) {
      int mpi_idx = Target.MPIIndex;
      if (mpi_idx >= 0 && mpi_idx < B.RequestCount && B.Requests[mpi_idx] != MPI_REQUEST_NULL) {
	mdmp_check_mpi(MPI_Wait(&B.Requests[mpi_idx], MPI_STATUS_IGNORE), "MPI_Wait(declarative logical)");
	B.Requests[mpi_idx] = MPI_REQUEST_NULL;
	mdmp_note_requests_completed(1);
        
	bool all_done = true;
	for (int i=0; i < B.RequestCount; i++) {
	  if (B.Requests[i] != MPI_REQUEST_NULL) { all_done = false; break; }
	}
	if (all_done) B.Active = false;
      }
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
  // Lock Bypass Fast Path
  if (mdmp_runtime_active.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(mdmp_mpi_mutex);
    mdmp_wait_unlocked(req_id);
  } else {
    mdmp_wait_unlocked(req_id);
  }
}

void mdmp_wait_many_sequential_unlocked(const int *uniq, int count) {
  for (int i = 0; i < count; ++i) {
    mdmp_wait_unlocked(uniq[i]);
  }
}

void mdmp_wait_many(const int *ids, int n) {
  if (ids == nullptr || n <= 0)
    return;

  uint64_t t0 = 0;
  if (mdmp_profile_enabled) {
    t0 = mdmp_now_us();
    mdmp_prof_wait_many_calls++;
    mdmp_prof_wait_many_ids_total += (uint64_t)n;
  }

  auto finish_wait_many_profile = [&](uint64_t *path_counter) {
    if (mdmp_profile_enabled) {
      if (path_counter)
        (*path_counter)++;
      mdmp_prof_wait_many_time_us += (mdmp_now_us() - t0);
    }
  };

  bool use_locks = mdmp_runtime_active.load(std::memory_order_relaxed);

  // --------------------------------------------------------------------------
  // 1. MICRO-STENCIL FAST PATH
  // --------------------------------------------------------------------------
  if (n <= 8 && !use_locks) {
    MPI_Request stack_reqs[8];
    int active_slots[8];
    int active_count = 0;
    bool fast_path_valid = true;

    for (int i = 0; i < n; ++i) {
      int id = ids[i];
      if (id == MDMP_PROCESS_NOT_INVOLVED)
        continue;

      if (id < 0 || id >= MDMP_MAX_REQUESTS) {
        fast_path_valid = false;
        break;
      }

      bool duplicate = false;
      for (int j = 0; j < active_count; ++j) {
        if (active_slots[j] == id) {
          duplicate = true;
          break;
        }
      }
      if (duplicate)
        continue;

      if (mdmp_request_pool[id] != MPI_REQUEST_NULL) {
        stack_reqs[active_count] = mdmp_request_pool[id];
        active_slots[active_count] = id;
        active_count++;
      }
    }

    if (fast_path_valid) {
      if (active_count > 0) {
        mdmp_check_mpi(
		       MPI_Waitall(active_count, stack_reqs, MPI_STATUSES_IGNORE),
		       "MPI_Waitall(fast path)");
        mdmp_note_requests_completed(active_count);

        for (int i = 0; i < active_count; ++i) {
          int slot = active_slots[i];
          mdmp_request_pool[slot] = MPI_REQUEST_NULL;
          mdmp_release_imperative_slot_unlocked(slot);
        }
      }

      finish_wait_many_profile(&mdmp_prof_wait_many_fast_calls);
      return;
    }
  }

  std::unique_lock<std::mutex> lock(mdmp_mpi_mutex, std::defer_lock);
  if (use_locks)
    lock.lock();

  // --------------------------------------------------------------------------
  // 2. TINY SEQUENTIAL DEDUP PATH
  // --------------------------------------------------------------------------
  if (n <= MDMP_WAIT_MANY_TINY_CUTOFF) {
    int uniq[MDMP_WAIT_MANY_TINY_CUTOFF];
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

      if (!seen)
        uniq[u++] = id;
    }

    mdmp_wait_many_sequential_unlocked(uniq, u);
    finish_wait_many_profile(&mdmp_prof_wait_many_tiny_seq_calls);
    return;
  }

  // --------------------------------------------------------------------------
  // 3. LARGE FALLBACK PATH
  // --------------------------------------------------------------------------
  for (int i = 0; i < n; ++i) {
    mdmp_wait_unlocked(ids[i]);
  }

  finish_wait_many_profile(&mdmp_prof_wait_many_fallback_calls);
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

  mdmp_imper_active_count = 0;
  mdmp_imper_free_top = MDMP_MAX_REQUESTS;

  for (int i = 0; i < MDMP_MAX_REQUESTS; ++i) {
    mdmp_request_pool[i] = MPI_REQUEST_NULL;
    mdmp_imper_free_slots[i] = MDMP_MAX_REQUESTS - 1 - i;
    mdmp_imper_active_pos[i] = -1;
  }

  mdmp_decl_req_counter = 0;
  mdmp_decl_batch_counter = 0;
  
  send_q_count = 0;
  recv_q_count = 0;
  reduce_q_count = 0;
  gather_q_count = 0;
  allreduce_q_count = 0;
  allgather_q_count = 0;
  bcast_q_count = 0;
  for (int i=0; i<16; i++) mdmp_batch_pool[i].Active = false;

  mdmp_active_request_count.store(0, std::memory_order_relaxed);

  for (int i = 0; i < MDMP_MAX_REQUESTS; ++i) mdmp_request_pool[i] = MPI_REQUEST_NULL;
  
  const char* env_debug = getenv("MDMP_DEBUG");
  if (env_debug != NULL && atoi(env_debug) > 0) mdmp_debug_enabled = 1;

  const char* env_prof = getenv("MDMP_PROFILE");
  if (env_prof != NULL && atoi(env_prof) > 0) mdmp_profile_enabled = 1;
  else mdmp_profile_enabled = 0;

  mdmp_profile_reset();

  mdmp_log("[MDMP Runtime] Starting up thread progress engine on rank %d\n", global_my_rank);
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
    if (mdmp_progress_thread.joinable())
      mdmp_progress_thread.join();
  }

  mdmp_wait_all_declarative_batches_unlocked();
  mdmp_profile_report();
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
  if (mdmp_runtime_active) lock.lock();
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

  uint64_t t0 = 0;
  if (mdmp_profile_enabled)
    t0 = mdmp_now_us();

  bool use_locks = mdmp_runtime_active.load(std::memory_order_relaxed);
  if (use_locks)
    mdmp_mpi_mutex.lock();

  int id = mdmp_alloc_imperative_slot_unlocked();
  int actual_count = mdmp_actual_count(count, type, bytes);
  uint64_t actual_bytes = mdmp_actual_bytes(count, type, bytes);

  mdmp_check_mpi(
		 MPI_Isend(buf, actual_count, get_mpi_type(type),
			   dest, tag, mdmp_comm, &mdmp_request_pool[id]),
		 "MPI_Isend");
  mdmp_note_requests_started(1);

  if (use_locks)
    mdmp_mpi_mutex.unlock();

  if (mdmp_profile_enabled) {
    mdmp_prof_send_calls++;
    mdmp_prof_send_bytes += actual_bytes;
    mdmp_prof_send_time_us += (mdmp_now_us() - t0);
  }

  return id;
}

int mdmp_recv(void* buf, size_t count, int type, size_t bytes, int receiver, int src, int tag) {
  if (src < 0 || src >= global_size || global_my_rank != receiver)
    return MDMP_PROCESS_NOT_INVOLVED;

  uint64_t t0 = 0;
  if (mdmp_profile_enabled)
    t0 = mdmp_now_us();

  bool use_locks = mdmp_runtime_active.load(std::memory_order_relaxed);
  if (use_locks)
    mdmp_mpi_mutex.lock();

  int id = mdmp_alloc_imperative_slot_unlocked();
  int actual_count = mdmp_actual_count(count, type, bytes);
  uint64_t actual_bytes = mdmp_actual_bytes(count, type, bytes);

  mdmp_check_mpi(
		 MPI_Irecv(buf, actual_count, get_mpi_type(type),
			   src, tag, mdmp_comm, &mdmp_request_pool[id]),
		 "MPI_Irecv");
  mdmp_note_requests_started(1);

  if (use_locks)
    mdmp_mpi_mutex.unlock();

  if (mdmp_profile_enabled) {
    mdmp_prof_recv_calls++;
    mdmp_prof_recv_bytes += actual_bytes;
    mdmp_prof_recv_time_us += (mdmp_now_us() - t0);
  }

  return id;
}

int mdmp_reduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes, int root_rank, int op) {
  bool use_locks = mdmp_runtime_active.load(std::memory_order_relaxed);
  if (use_locks) mdmp_mpi_mutex.lock();

  int id = mdmp_alloc_imperative_slot_unlocked();
  const void* final_sendbuf = (sendbuf == recvbuf && global_my_rank == root_rank) ? MPI_IN_PLACE : sendbuf;
  int actual_count = mdmp_actual_count(count, type, bytes);
  mdmp_check_mpi(MPI_Ireduce(final_sendbuf, recvbuf, actual_count, get_mpi_type(type), get_mpi_op(op), root_rank, mdmp_comm, &mdmp_request_pool[id]), "MPI_Ireduce");
  mdmp_note_requests_started(1);

  if (use_locks) mdmp_mpi_mutex.unlock();
  return id;
}

int mdmp_gather(void* sendbuf, size_t sendcount, void* recvbuf, int type, size_t bytes, int root_rank) {
  bool use_locks = mdmp_runtime_active.load(std::memory_order_relaxed);
  if (use_locks) mdmp_mpi_mutex.lock();

  int id = mdmp_alloc_imperative_slot_unlocked();
  const void* final_sb = (sendbuf == recvbuf && global_my_rank == root_rank) ? MPI_IN_PLACE : sendbuf;
  int actual_count = mdmp_actual_count(sendcount, type, bytes);
  mdmp_check_mpi(MPI_Igather(final_sb, actual_count, get_mpi_type(type), recvbuf, actual_count, get_mpi_type(type), root_rank, mdmp_comm, &mdmp_request_pool[id]), "MPI_Igather");
  mdmp_note_requests_started(1);

  if (use_locks) mdmp_mpi_mutex.unlock();
  return id;
}

int mdmp_allreduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes, int op) {
  bool use_locks = mdmp_runtime_active.load(std::memory_order_relaxed);
  if (use_locks) mdmp_mpi_mutex.lock();

  int id = mdmp_alloc_imperative_slot_unlocked();
  const void* final_sendbuf = (sendbuf == recvbuf) ? MPI_IN_PLACE : sendbuf;
  int actual_count = mdmp_actual_count(count, type, bytes);
  mdmp_check_mpi(MPI_Iallreduce(final_sendbuf, recvbuf, actual_count, get_mpi_type(type), get_mpi_op(op), mdmp_comm, &mdmp_request_pool[id]), "MPI_Iallreduce");
  mdmp_note_requests_started(1);

  if (use_locks) mdmp_mpi_mutex.unlock();
  return id;
}

int mdmp_allgather(void* sendbuf, size_t count, void* recvbuf, int type, size_t bytes) {
  bool use_locks = mdmp_runtime_active.load(std::memory_order_relaxed);
  if (use_locks) mdmp_mpi_mutex.lock();

  int id = mdmp_alloc_imperative_slot_unlocked();
  const void* final_sb = (sendbuf == recvbuf) ? MPI_IN_PLACE : sendbuf;
  int actual_count = mdmp_actual_count(count, type, bytes);
  mdmp_check_mpi(MPI_Iallgather(final_sb, actual_count, get_mpi_type(type), recvbuf, actual_count, get_mpi_type(type), mdmp_comm, &mdmp_request_pool[id]), "MPI_Iallgather");
  mdmp_note_requests_started(1);

  if (use_locks) mdmp_mpi_mutex.unlock();
  return id;
}

int mdmp_bcast(void* buffer, size_t count, int type, size_t bytes, int root_rank) {
  bool use_locks = mdmp_runtime_active.load(std::memory_order_relaxed);
  if (use_locks) mdmp_mpi_mutex.lock();

  int id = mdmp_alloc_imperative_slot_unlocked();
  int actual_count = mdmp_actual_count(count, type, bytes);
  mdmp_check_mpi(MPI_Ibcast(buffer, actual_count, get_mpi_type(type), root_rank, mdmp_comm, &mdmp_request_pool[id]), "MPI_Ibcast");
  mdmp_note_requests_started(1);

  if (use_locks) mdmp_mpi_mutex.unlock();
  return id;
}

// ==============================================================================
// Declarative Registration (Static Queues)
// ==============================================================================

int mdmp_register_send(void* buffer, size_t count, int type, size_t bytes, int sender_rank, int dest_rank, int tag) {
  if (dest_rank >= 0 && dest_rank < global_size && global_my_rank == sender_rank) {
    if (send_q_count >= MDMP_STATIC_Q_SIZE) mdmp_abort(1);
    uint32_t id = mdmp_decl_req_counter++;
    send_queue[send_q_count++] = {buffer, count, type, bytes, dest_rank, tag, (int)id}; 
    return (int)id + MDMP_MAX_REQUESTS;
  }
  return MDMP_PROCESS_NOT_INVOLVED;
}

int mdmp_register_recv(void* buffer, size_t count, int type, size_t bytes, int receiver_rank, int src_rank, int tag) {  
  if (src_rank >= 0 && src_rank < global_size && global_my_rank == receiver_rank) {
    if (recv_q_count >= MDMP_STATIC_Q_SIZE) mdmp_abort(1);
    uint32_t id = mdmp_decl_req_counter++;
    recv_queue[recv_q_count++] = {buffer, count, type, bytes, src_rank, tag, (int)id}; 
    return (int)id + MDMP_MAX_REQUESTS;
  }
  return MDMP_PROCESS_NOT_INVOLVED;
}

int mdmp_register_reduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes, int root_rank, int op) {
  if (reduce_q_count >= MDMP_STATIC_Q_SIZE) mdmp_abort(1);
  uint32_t id = mdmp_decl_req_counter++; 
  reduce_queue[reduce_q_count++] = {sendbuf, recvbuf, count, type, bytes, root_rank, op, (int)id};
  return (int)id + MDMP_MAX_REQUESTS;
}

int mdmp_register_gather(void* sendbuf, size_t sendcount, void* recvbuf, int type, size_t bytes, int root_rank) {
  if (gather_q_count >= MDMP_STATIC_Q_SIZE) mdmp_abort(1);
  uint32_t id = mdmp_decl_req_counter++;
  gather_queue[gather_q_count++] = {sendbuf, sendcount, recvbuf, type, bytes, root_rank, (int)id};
  return (int)id + MDMP_MAX_REQUESTS;
}

int mdmp_register_allreduce(void* sendbuf, void* recvbuf, size_t count, int type,  size_t bytes, int op) {
  if (allreduce_q_count >= MDMP_STATIC_Q_SIZE) mdmp_abort(1);
  uint32_t id = mdmp_decl_req_counter++;
  allreduce_queue[allreduce_q_count++] = {sendbuf, recvbuf, count, type, bytes, op, (int)id};
  return (int)id + MDMP_MAX_REQUESTS;
}

int mdmp_register_allgather(void* sendbuf, size_t count, void* recvbuf, int type, size_t bytes) {
  if (allgather_q_count >= MDMP_STATIC_Q_SIZE) mdmp_abort(1);
  uint32_t id = mdmp_decl_req_counter++;
  allgather_queue[allgather_q_count++] = {sendbuf, count, recvbuf, type, bytes, (int)id};
  return (int)id + MDMP_MAX_REQUESTS;
}

int mdmp_register_bcast(void* buffer, size_t count, int type, size_t bytes, int root_rank) {
  if (bcast_q_count >= MDMP_STATIC_Q_SIZE) mdmp_abort(1);
  uint32_t id = mdmp_decl_req_counter++;
  bcast_queue[bcast_q_count++] = {buffer, count, type, bytes, root_rank, (int)id};
  return (int)id + MDMP_MAX_REQUESTS;
}

int mdmp_commit() {
  bool use_locks = mdmp_runtime_active.load(std::memory_order_relaxed);
  if (use_locks) mdmp_mpi_mutex.lock();

  DeclarativeBatch& Batch = mdmp_batch_pool[mdmp_decl_batch_counter % 16];
  Batch.Serial = mdmp_decl_batch_counter++;
  Batch.RequestCount = 0;
  Batch.Active = true;

  auto MapLogicalReq = [&](int logicalReqID, int mpi_idx) {
    decl_to_req_target[logicalReqID % 65536] = {Batch.Serial, mpi_idx};
  };

  // Zero-Allocation Direct Dispatch
  for (int i = 0; i < recv_q_count; ++i) {
    int actual = (recv_queue[i].type == 4) ? recv_queue[i].bytes : recv_queue[i].count;
    mdmp_check_mpi(MPI_Irecv(recv_queue[i].buffer, actual, get_mpi_type(recv_queue[i].type), 
			     recv_queue[i].rank, recv_queue[i].tag, mdmp_comm, &Batch.Requests[Batch.RequestCount]), "MPI_Irecv");
    MapLogicalReq(recv_queue[i].req_id, Batch.RequestCount);
    Batch.RequestCount++;
  }
  recv_q_count = 0;

  for (int i = 0; i < send_q_count; ++i) {
    int actual = (send_queue[i].type == 4) ? send_queue[i].bytes : send_queue[i].count;
    mdmp_check_mpi(MPI_Isend(send_queue[i].buffer, actual, get_mpi_type(send_queue[i].type), 
			     send_queue[i].rank, send_queue[i].tag, mdmp_comm, &Batch.Requests[Batch.RequestCount]), "MPI_Isend");
    MapLogicalReq(send_queue[i].req_id, Batch.RequestCount);
    Batch.RequestCount++;
  }
  send_q_count = 0;

  for (int i = 0; i < reduce_q_count; ++i) {
    int actual = (reduce_queue[i].type == 4) ? reduce_queue[i].bytes : reduce_queue[i].count;
    const void* final_sb = (reduce_queue[i].sendbuf == reduce_queue[i].recvbuf && global_my_rank == reduce_queue[i].root) ? MPI_IN_PLACE : reduce_queue[i].sendbuf;
    mdmp_check_mpi(MPI_Ireduce(final_sb, reduce_queue[i].recvbuf, actual, get_mpi_type(reduce_queue[i].type), 
			       get_mpi_op(reduce_queue[i].op), reduce_queue[i].root, mdmp_comm, &Batch.Requests[Batch.RequestCount]), "MPI_Ireduce");
    MapLogicalReq(reduce_queue[i].req_id, Batch.RequestCount);
    Batch.RequestCount++;
  }
  reduce_q_count = 0;
  
  for (int i = 0; i < gather_q_count; ++i) {
    int actual = (gather_queue[i].type == 4) ? gather_queue[i].bytes : gather_queue[i].sendcount;
    const void* final_sb = (gather_queue[i].sendbuf == gather_queue[i].recvbuf && global_my_rank == gather_queue[i].root) ? MPI_IN_PLACE : gather_queue[i].sendbuf;
    mdmp_check_mpi(MPI_Igather(final_sb, actual, get_mpi_type(gather_queue[i].type), gather_queue[i].recvbuf, actual, get_mpi_type(gather_queue[i].type), gather_queue[i].root, mdmp_comm, &Batch.Requests[Batch.RequestCount]), "MPI_Igather");
    MapLogicalReq(gather_queue[i].req_id, Batch.RequestCount);
    Batch.RequestCount++;
  }
  gather_q_count = 0;

  for (int i = 0; i < allreduce_q_count; ++i) {
    int actual = (allreduce_queue[i].type == 4) ? allreduce_queue[i].bytes : allreduce_queue[i].count;
    const void* final_sb = (allreduce_queue[i].sendbuf == allreduce_queue[i].recvbuf) ? MPI_IN_PLACE : allreduce_queue[i].sendbuf;
    mdmp_check_mpi(MPI_Iallreduce(final_sb, allreduce_queue[i].recvbuf, actual, get_mpi_type(allreduce_queue[i].type), get_mpi_op(allreduce_queue[i].op), mdmp_comm, &Batch.Requests[Batch.RequestCount]), "MPI_Iallreduce");
    MapLogicalReq(allreduce_queue[i].req_id, Batch.RequestCount);
    Batch.RequestCount++;
  }
  allreduce_q_count = 0;

  for (int i = 0; i < allgather_q_count; ++i) {
    int actual = (allgather_queue[i].type == 4) ? allgather_queue[i].bytes : allgather_queue[i].count;
    const void* final_sb = (allgather_queue[i].sendbuf == allgather_queue[i].recvbuf) ? MPI_IN_PLACE : allgather_queue[i].sendbuf;
    mdmp_check_mpi(MPI_Iallgather(final_sb, actual, get_mpi_type(allgather_queue[i].type), allgather_queue[i].recvbuf, actual, get_mpi_type(allgather_queue[i].type), mdmp_comm, &Batch.Requests[Batch.RequestCount]), "MPI_Iallgather");
    MapLogicalReq(allgather_queue[i].req_id, Batch.RequestCount);
    Batch.RequestCount++;
  }
  allgather_q_count = 0;

  for (int i = 0; i < bcast_q_count; ++i) {
    int actual = (bcast_queue[i].type == 4) ? bcast_queue[i].bytes : bcast_queue[i].count;
    mdmp_check_mpi(MPI_Ibcast(bcast_queue[i].buffer, actual, get_mpi_type(bcast_queue[i].type), bcast_queue[i].root, mdmp_comm, &Batch.Requests[Batch.RequestCount]), "MPI_Ibcast");
    MapLogicalReq(bcast_queue[i].req_id, Batch.RequestCount);
    Batch.RequestCount++;
  }
  bcast_q_count = 0;

  if (Batch.RequestCount == 0) {
    Batch.Active = false;
    if (use_locks) mdmp_mpi_mutex.unlock();
    return MDMP_PROCESS_NOT_INVOLVED;
  }

  mdmp_note_requests_started(Batch.RequestCount);

  if (use_locks) mdmp_mpi_mutex.unlock();
  return mdmp_make_decl_batch_token(Batch.Serial);
}
