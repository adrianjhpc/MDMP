#ifndef MDMP_PRAGMA_INTERFACE_H
#define MDMP_PRAGMA_INTERFACE_H

#include <stddef.h>
#include <type_traits> // Required for decltype manipulation

#ifdef __cplusplus
#define MDMP_NOEXCEPT noexcept
#else
#define MDMP_NOEXCEPT
#endif

#ifdef __cplusplus

// 1. Our Internal Enum
enum MDMP_Datatype { MDMP_INT, MDMP_FLOAT, MDMP_DOUBLE, MDMP_CHAR };

// 2. Type Traits
template<typename T> struct MDMPTypeTraits;
template<> struct MDMPTypeTraits<int> { static const int type = MDMP_INT; };
template<> struct MDMPTypeTraits<float> { static const int type = MDMP_FLOAT; };
template<> struct MDMPTypeTraits<double> { static const int type = MDMP_DOUBLE; };
template<> struct MDMPTypeTraits<char> { static const int type = MDMP_CHAR; };

extern "C" {
#endif

void __mdmp_marker_init() MDMP_NOEXCEPT;
void __mdmp_marker_final() MDMP_NOEXCEPT;

int __mdmp_marker_get_size() MDMP_NOEXCEPT;
int __mdmp_marker_get_rank() MDMP_NOEXCEPT;

double __mdmp_marker_wtime() MDMP_NOEXCEPT;

void mdmp_set_debug(int enable) MDMP_NOEXCEPT;

void __mdmp_marker_commregion_begin() MDMP_NOEXCEPT;
void __mdmp_marker_commregion_end() MDMP_NOEXCEPT;
void __mdmp_marker_sync() MDMP_NOEXCEPT;

int __mdmp_marker_send(void* buffer, size_t count, int type, size_t byte_size, int sender, int dest, int tag) MDMP_NOEXCEPT;
int __mdmp_marker_recv(void* buffer, size_t count, int type, size_t byte_size, int receiver, int src, int tag) MDMP_NOEXCEPT;

int __mdmp_marker_reduce(void* in_buf, void* out_buf, size_t count, int type, size_t byte_size, int root, int op) MDMP_NOEXCEPT;

#ifdef __cplusplus
} // End extern "C"

#define MDMP_SEND(buf, count, sender, dest, tag) \
    __mdmp_marker_send((void*)(buf), count, MDMPTypeTraits<typename std::remove_reference<decltype(*(buf))>::type>::type, (count) * sizeof(*(buf)), sender, dest, tag)

#define MDMP_RECV(buf, count, receiver, src, tag) \
    __mdmp_marker_recv((void*)(buf), count, MDMPTypeTraits<typename std::remove_reference<decltype(*(buf))>::type>::type, (count) * sizeof(*(buf)), receiver, src, tag)

#define MDMP_REDUCE(in_buf, out_buf, count, root, op) \
    __mdmp_marker_reduce((void*)(in_buf), (void*)(out_buf), count, \
    MDMPTypeTraits<typename std::remove_reference<decltype(*(in_buf))>::type>::type, \
    (count) * sizeof(*(in_buf)), root, op)

#define MDMP_COMM_INIT()        __mdmp_marker_init()
#define MDMP_COMM_FINAL()       __mdmp_marker_final()
#define MDMP_COMMREGION_BEGIN() __mdmp_marker_commregion_begin()
#define MDMP_COMMREGION_END()   __mdmp_marker_commregion_end()
#define MDMP_COMM_SYNC()        __mdmp_marker_sync()

#define MDMP_GET_SIZE()         __mdmp_marker_get_size()
#define MDMP_GET_RANK()         __mdmp_marker_get_rank()
#define MDMP_WTIME()            __mdmp_marker_wtime()

#define MDMP_SET_DEBUG(enable)        mdmp_set_debug(enable)

#define MDMP_IGNORE -2
#define MDMP_RANK   __mdmp_marker_get_rank()

// Collective operation types (i.e. reduce)
#define MDMP_SUM 0
#define MDMP_MAX 1
#define MDMP_MIN 2

#endif
#endif
