#ifndef MDMP_INTERFACE_H
#define MDMP_INTERFACE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
    #include <type_traits>
    #define MDMP_NOEXCEPT noexcept

    // Default to MPI_BYTE if an unknown type is passed
    template<typename T> struct MDMPTypeTraits { static const int type = 4; };
    
    template<> struct MDMPTypeTraits<int>    { static const int type = 0; };
    template<> struct MDMPTypeTraits<double> { static const int type = 1; };
    template<> struct MDMPTypeTraits<float>  { static const int type = 2; };
    template<> struct MDMPTypeTraits<char>   { static const int type = 3; };
    template<> struct MDMPTypeTraits<int64_t> { static const int type = 5; };
    template<> struct MDMPTypeTraits<long long> { static const int type = 5; }; 

    #define MDMP_DEDUCE_TYPE(ptr) MDMPTypeTraits<typename std::remove_reference<decltype(*(ptr))>::type>::type

    extern "C" {
#else
    #define MDMP_NOEXCEPT

    #define MDMP_DEDUCE_TYPE(ptr) _Generic((*(ptr)), \
        int: 0, \
        double: 1, \
        float: 2, \
        char: 3, \
        int64_t: 5, \
        long long: 5, \
        default: 4 \
    )

#endif
    void __mdmp_marker_init() MDMP_NOEXCEPT;
    void __mdmp_marker_final() MDMP_NOEXCEPT;
    int __mdmp_marker_get_size() MDMP_NOEXCEPT;
    int __mdmp_marker_get_rank() MDMP_NOEXCEPT;
    void __mdmp_marker_commregion_begin() MDMP_NOEXCEPT;
    void __mdmp_marker_commregion_end() MDMP_NOEXCEPT;
    void __mdmp_marker_sync() MDMP_NOEXCEPT;
    double __mdmp_marker_wtime() MDMP_NOEXCEPT;
    void __mdmp_marker_set_debug(int enable) MDMP_NOEXCEPT;
    void __mdmp_marker_abort(int error_code) MDMP_NOEXCEPT;

    int __mdmp_marker_send(void* buffer, size_t count, int type, size_t byte_size, int sender, int dest, int tag) MDMP_NOEXCEPT;
    int __mdmp_marker_recv(void* buffer, size_t count, int type, size_t byte_size, int receiver, int src, int tag) MDMP_NOEXCEPT;

    int __mdmp_marker_reduce(void* in_buf, void* out_buf, size_t count, int type, size_t byte_size, int root, int op) MDMP_NOEXCEPT;
    int __mdmp_marker_gather(void* send_buf, size_t send_count, void* recv_buf, int type, size_t byte_size, int root) MDMP_NOEXCEPT;
      
    int __mdmp_marker_allreduce(void* in_buf, void* out_buf, size_t count, int type, size_t byte_size, int op) MDMP_NOEXCEPT;
    int __mdmp_marker_allgather(void* in_buf, size_t count, void* out_buf, int type, size_t byte_size) MDMP_NOEXCEPT;

    int __mdmp_marker_bcast(void* buffer, size_t count, int type, size_t byte_size, int root) MDMP_NOEXCEPT;
      
    int __mdmp_marker_register_send(void* buffer, size_t count, int type, size_t byte_size, int sender, int dest, int tag) MDMP_NOEXCEPT;
    int __mdmp_marker_register_recv(void* buffer, size_t count, int type, size_t byte_size, int receiver, int src, int tag) MDMP_NOEXCEPT;

    int __mdmp_marker_register_reduce(void* in_buf, void* out_buf, size_t count, int type, size_t byte_size, int root, int op) MDMP_NOEXCEPT;
    int __mdmp_marker_register_gather(void* send_buf, size_t send_count, void* recv_buf, int type, size_t byte_size, int root) MDMP_NOEXCEPT;

    int __mdmp_marker_register_allreduce(void* in_buf, void* out_buf, size_t count, int type, size_t byte_size, int op) MDMP_NOEXCEPT;
    int __mdmp_marker_register_allgather(void* in_buf, size_t count, void* out_buf, int type, size_t byte_size) MDMP_NOEXCEPT;

    int __mdmp_marker_register_bcast(void* buffer, size_t count, int type, size_t byte_size, int root) MDMP_NOEXCEPT;
      
    int __mdmp_marker_commit() MDMP_NOEXCEPT;

#ifdef __cplusplus
    }
#endif

#define MDMP_IGNORE -2
#define MDMP_SUM 0
#define MDMP_MAX 1
#define MDMP_MIN 2
#define MDMP_PROD 3

#define MDMP_SEND(buf, count, sender, dest, tag) \
    __mdmp_marker_send((void*)(buf), count, MDMP_DEDUCE_TYPE(buf), (count) * sizeof(*(buf)), sender, dest, tag)

#define MDMP_RECV(buf, count, receiver, src, tag) \
    __mdmp_marker_recv((void*)(buf), count, MDMP_DEDUCE_TYPE(buf), (count) * sizeof(*(buf)), receiver, src, tag)

#define MDMP_REDUCE(in_buf, out_buf, count, root, op) \
    __mdmp_marker_reduce((void*)(in_buf), (void*)(out_buf), count, MDMP_DEDUCE_TYPE(in_buf), (count) * sizeof(*(in_buf)), root, op)

#define MDMP_GATHER(send_buf, send_count, recv_buf, root) \
    __mdmp_marker_gather((void*)(send_buf), send_count, (void*)(recv_buf), MDMP_DEDUCE_TYPE(send_buf), (send_count) * sizeof(*(send_buf)), root)

#define MDMP_ALLREDUCE(in_buf, out_buf, count, op) \
    __mdmp_marker_allreduce((void*)(in_buf), (void*)(out_buf), count, MDMP_DEDUCE_TYPE(in_buf), (count) * sizeof(*(in_buf)), op)

#define MDMP_ALLGATHER(in_buf, count, out_buf) \
    __mdmp_marker_allgather((void*)(in_buf), count, (void*)(out_buf), MDMP_DEDUCE_TYPE(in_buf), (count) * sizeof(*(in_buf)))

#define MDMP_BCAST(buf, count, root) \
    __mdmp_marker_bcast((void*)(buf), count, MDMP_DEDUCE_TYPE(buf), (count) * sizeof(*(buf)), root)

#define MDMP_REGISTER_SEND(buf, count, sender, dest, tag) \
    __mdmp_marker_register_send((void*)(buf), count, MDMP_DEDUCE_TYPE(buf), (count) * sizeof(*(buf)), sender, dest, tag)

#define MDMP_REGISTER_RECV(buf, count, receiver, src, tag) \
    __mdmp_marker_register_recv((void*)(buf), count, MDMP_DEDUCE_TYPE(buf), (count) * sizeof(*(buf)), receiver, src, tag)

#define MDMP_REGISTER_REDUCE(in_buf, out_buf, count, root, op) \
    __mdmp_marker_register_reduce((void*)(in_buf), (void*)(out_buf), count, MDMP_DEDUCE_TYPE(in_buf), (count) * sizeof(*(in_buf)), root, op)

#define MDMP_REGISTER_GATHER(send_buf, send_count, recv_buf, root) \
    __mdmp_marker_register_gather((void*)(send_buf), send_count, (void*)(recv_buf), MDMP_DEDUCE_TYPE(send_buf), (send_count) * sizeof(*(send_buf)), root)

#define MDMP_REGISTER_ALLREDUCE(in_buf, out_buf, count, op) \
    __mdmp_marker_register_allreduce((void*)(in_buf), (void*)(out_buf), count, MDMP_DEDUCE_TYPE(in_buf), (count) * sizeof(*(in_buf)), op)

#define MDMP_REGISTER_ALLGATHER(in_buf, count, out_buf) \
    __mdmp_marker_register_allgather((void*)(in_buf), count, (void*)(out_buf), MDMP_DEDUCE_TYPE(in_buf), (count) * sizeof(*(in_buf)))

#define MDMP_REGISTER_BCAST(buf, count, root) \
    __mdmp_marker_register_bcast((void*)(buf), count, MDMP_DEDUCE_TYPE(buf), (count) * sizeof(*(buf)), root)

#define MDMP_COMMIT() __mdmp_marker_commit()

#define MDMP_COMM_INIT()        __mdmp_marker_init()
#define MDMP_COMM_FINAL()       __mdmp_marker_final()
#define MDMP_COMMREGION_BEGIN() __mdmp_marker_commregion_begin()
#define MDMP_COMMREGION_END()   __mdmp_marker_commregion_end()
#define MDMP_COMM_SYNC()        __mdmp_marker_sync()

#define MDMP_GET_SIZE()         __mdmp_marker_get_size()
#define MDMP_GET_RANK()         __mdmp_marker_get_rank()
#define MDMP_RANK               __mdmp_marker_get_rank() // Alias for cleaner maths
#define MDMP_WTIME()            __mdmp_marker_wtime()

#define MDMP_ABORT(error_code) __mdmp_marker_abort(error_code)

#define MDMP_SET_DEBUG(enable)  __mdmp_marker_set_debug(enable)

#endif // MDMP_INTERFACE_H
