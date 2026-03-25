// mdmp_runtime.h
#ifndef MDMP_RUNTIME_H
#define MDMP_RUNTIME_H

#include <stddef.h>
#include <stdlib.h> // For getenv
#include <stdarg.h> // For variadic arguments
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// MDMP runtime function declarations
void mdmp_log(const char* format, ...);
void mdmp_set_debug(int enable) noexcept;
void mdmp_init();
void mdmp_final();
void mdmp_commregion_begin();
void mdmp_commregion_end();
int mdmp_send(void* buffer, size_t count, int type, int sender, int dest, int tag);
int mdmp_recv(void* buffer, size_t count, int type, int receiver, int src, int tag);
void mdmp_sync();
void mdmp_wait(int req_id);
int mdmp_get_size();
int mdmp_get_rank();
double mdmp_wtime();
void mdmp_optimize(int level);
void mdmp_no_opt();
int mdmp_reduce(void* in_buf, void* out_buf, size_t count, int type, int root, int op);

#ifdef __cplusplus
}
#endif

#endif // MDMP_RUNTIME_H

