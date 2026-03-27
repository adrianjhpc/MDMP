#ifndef MDMP_RUNTIME_H
#define MDMP_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- LIFECYCLE & UTILITIES ---
void mdmp_init();
void mdmp_final();
int  mdmp_get_rank();
int  mdmp_get_size();
double mdmp_wtime();
void mdmp_sync();

// --- REGION MARKERS ---
void mdmp_commregion_begin();
void mdmp_commregion_end();

// --- UNIFIED WAIT ENGINE ---
// Accepts a specific request ID (Imperative) or -1 for a bulk Waitall (Declarative)
void mdmp_wait(int req_id);

// ========================================================================
// PARADIGM 1: IMPERATIVE API (Fine-grained, returns individual Request IDs)
// ========================================================================
int mdmp_send(void* buffer, size_t count, int type, int sender_rank, int dest_rank, int tag);
int mdmp_recv(void* buffer, size_t count, int type, int receiver_rank, int src_rank, int tag);

int mdmp_reduce(void* sendbuf, void* recvbuf, size_t count, int type, int root_rank, int op);
int mdmp_gather(void* sendbuf, size_t sendcount, void* recvbuf, int type, int root_rank);

// ========================================================================
// PARADIGM 2: DECLARATIVE API (Inspector-Executor, returns -1 Batch ID)
// ========================================================================
void mdmp_register_send(void* buffer, size_t count, int type, int sender_rank, int dest_rank, int tag);
void mdmp_register_recv(void* buffer, size_t count, int type, int receiver_rank, int src_rank, int tag);
int  mdmp_commit();

#ifdef __cplusplus
}
#endif

#endif // MDMP_RUNTIME_H
