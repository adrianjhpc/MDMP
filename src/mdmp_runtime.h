// mdmp_runtime.h
#ifndef MDMP_RUNTIME_H
#define MDMP_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

// MDMP runtime function declarations
void mdmp_commregion_begin();
void mdmp_commregion_end();
void mdmp_sync();
void mdmp_wait();
void mdmp_send();
void mdmp_recv();
void mdmp_get_size();
void mdmp_get_rank();
void mdmp_optimize(int level);
void mdmp_no_opt();
void mdmp_reduce(int op, void* src, void* dst, size_t size);
void mdmp_barrier();
void mdmp_broadcast(void* data, size_t size, int root);
void mdmp_gather(void* sendbuf, void* recvbuf, int count, int datatype, int root);
void mdmp_scatter(void* sendbuf, void* recvbuf, int count, int datatype, int root);

#ifdef __cplusplus
}
#endif

#endif // MDMP_RUNTIME_H

