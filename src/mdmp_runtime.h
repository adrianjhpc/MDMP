#ifndef MDMP_RUNTIME_H
#define MDMP_RUNTIME_H

#include <stddef.h>
#include <string.h> 
#include <stdlib.h>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>

#ifdef __cplusplus
extern "C" {
#endif

#define MDMP_MAX_REQUESTS 8192
#define MDMP_MAX_DECLARATIVE_REQS 1024
#define MDMP_DECLARATIVE_WAIT -1
#define MDMP_PROCESS_NOT_INVOLVED -2
  
  void mdmp_init();
  void mdmp_final();
  int  mdmp_get_rank();
  int  mdmp_get_size();
  double mdmp_wtime();
  void mdmp_sync();
  void mdmp_set_debug(int enabled);
  void mdmp_commregion_begin();
  void mdmp_commregion_end();

  void mdmp_progress();

  void mdmp_abort(int error_code);

  // Accepts a specific request ID (Imperative) or -1 for a bulk Waitall (Declarative)
  void mdmp_wait(int req_id);

  // ========================================================================
  // Imperative API (Fine-grained, returns individual Request IDs)
  // ========================================================================
  int mdmp_send(void* buf, size_t count, int type, size_t bytes, int sender, int dest, int tag);
  int mdmp_recv(void* buf, size_t count, int type, size_t bytes, int receiver, int src, int tag);

  int mdmp_reduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes, int root_rank, int op);
  int mdmp_gather(void* sendbuf, size_t sendcount, void* recvbuf, int type, size_t bytes,int root_rank);

  int mdmp_allreduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes,int op);
  int mdmp_allgather(void* sendbuf, size_t count, void* recvbuf, int type, size_t bytes);

  int mdmp_bcast(void* buffer, size_t count, int type, size_t bytes, int root);
  
  // ========================================================================
  // Declarative API (Inspector-Executor, returns -1 Batch ID)
  // ========================================================================
  int mdmp_register_send(void* buffer, size_t count, int type, size_t bytes, int sender_rank, int dest_rank, int tag);
  int mdmp_register_recv(void* buffer, size_t count, int type, size_t bytes, int receiver_rank, int src_rank, int tag);

  int mdmp_register_reduce(void* sendbuf, void* recvbuf, size_t count, int type, size_t bytes, int root_rank, int op);
  int mdmp_register_gather(void* sendbuf, size_t sendcount, void* recvbuf, int type, size_t bytes, int root_rank);

  int mdmp_register_allreduce(void* sendbuf, void* recvbuf, size_t count, int type,  size_t bytes, int op);
  int mdmp_register_allgather(void* sendbuf, size_t count, void* recvbuf, int type, size_t bytes);

  int mdmp_register_bcast(void* buffer, size_t count, int type, size_t bytes, int root_rank);

  int  mdmp_commit();

#ifdef __cplusplus
}
#endif

// Dynamic logging macro
#define mdmp_log(...) do {			\
    if (mdmp_debug_enabled) {			\
      printf(__VA_ARGS__);			\
      fflush(stdout);				\
    }						\
  } while(0)

#endif // MDMP_RUNTIME_H
