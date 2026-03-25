#ifndef MDMP_PRAGMA_INTERFACE_H
#define MDMP_PRAGMA_INTERFACE_H

#include <stddef.h>
#include <type_traits> // Required for decltype manipulation

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

void __mdmp_marker_init();
void __mdmp_marker_final();

int __mdmp_marker_get_size();
int __mdmp_marker_get_rank();

void __mdmp_marker_commregion_begin();
void __mdmp_marker_commregion_end();
void __mdmp_marker_sync();

int __mdmp_marker_send(void* buffer, size_t count, int type, size_t byte_size, int sender, int dest);
int __mdmp_marker_recv(void* buffer, size_t count, int type, size_t byte_size, int receiver, int src);

#ifdef __cplusplus
} // End extern "C"

#define MDMP_SEND(buf, count, sender, dest) \
    __mdmp_marker_send((void*)(buf), count, MDMPTypeTraits<typename std::remove_reference<decltype(*(buf))>::type>::type, (count) * sizeof(*(buf)), sender, dest)

#define MDMP_RECV(buf, count, receiver, src) \
    __mdmp_marker_recv((void*)(buf), count, MDMPTypeTraits<typename std::remove_reference<decltype(*(buf))>::type>::type, (count) * sizeof(*(buf)), receiver, src)

#define MDMP_COMM_INIT()        __mdmp_marker_init()
#define MDMP_COMM_FINAL()       __mdmp_marker_final()
#define MDMP_COMMREGION_BEGIN() __mdmp_marker_commregion_begin()
#define MDMP_COMMREGION_END()   __mdmp_marker_commregion_end()
#define MDMP_COMM_SYNC()        __mdmp_marker_sync()

#define MDMP_GET_SIZE()         __mdmp_marker_get_size()
#define MDMP_GET_RANK()         __mdmp_marker_get_rank()

#endif
#endif
