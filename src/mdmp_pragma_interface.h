#ifndef MDMP_PRAGMA_INTERFACE_H
#define MDMP_PRAGMA_INTERFACE_H

#include <stddef.h>

// ==============================================================================
// Language-Specific Type Deduction
// ==============================================================================
#ifdef __cplusplus
    // --- C++ PATH ---
    #include <type_traits>
    #define MDMP_NOEXCEPT noexcept

    // Your existing C++ type traits
    template<typename T> struct MDMPTypeTraits;
    template<> struct MDMPTypeTraits<int>    { static const int type = 0; };
    template<> struct MDMPTypeTraits<double> { static const int type = 1; };
    template<> struct MDMPTypeTraits<float>  { static const int type = 2; };
    template<> struct MDMPTypeTraits<char>   { static const int type = 3; };

    // A unified helper macro for C++
    #define MDMP_DEDUCE_TYPE(ptr) MDMPTypeTraits<typename std::remove_reference<decltype(*(ptr))>::type>::type

    // C++ requires extern "C" to prevent name mangling of our dummy markers
    extern "C" {
#else
    // --- PURE C PATH ---
    #define MDMP_NOEXCEPT 

    // C11 _Generic allows compile-time type deduction in pure C!
    #define MDMP_DEDUCE_TYPE(ptr) _Generic((*(ptr)), \
        int: 0, \
        double: 1, \
        float: 2, \
        char: 3, \
        default: 0 \
    )
#endif

// ==============================================================================
// Dummy Marker Declarations (Shared by C and C++)
// ==============================================================================
    void __mdmp_marker_init() MDMP_NOEXCEPT;
    void __mdmp_marker_final() MDMP_NOEXCEPT;
    int __mdmp_marker_get_size() MDMP_NOEXCEPT;
    int __mdmp_marker_get_rank() MDMP_NOEXCEPT;
    void __mdmp_marker_commregion_begin() MDMP_NOEXCEPT;
    void __mdmp_marker_commregion_end() MDMP_NOEXCEPT;
    void __mdmp_marker_sync() MDMP_NOEXCEPT;
    double __mdmp_marker_wtime() MDMP_NOEXCEPT;
    void mdmp_set_debug(int enable) MDMP_NOEXCEPT;
    
    int __mdmp_marker_send(void* buffer, size_t count, int type, size_t byte_size, int sender, int dest, int tag) MDMP_NOEXCEPT;
    int __mdmp_marker_recv(void* buffer, size_t count, int type, size_t byte_size, int receiver, int src, int tag) MDMP_NOEXCEPT;

    int __mdmp_marker_reduce(void* in_buf, void* out_buf, size_t count, int type, size_t byte_size, int root, int op) MDMP_NOEXCEPT;
    int __mdmp_marker_gather(void* send_buf, size_t send_count, void* recv_buf, int type, size_t byte_size, int root) MDMP_NOEXCEPT;

#ifdef __cplusplus
    } // Close extern "C" block for C++
#endif

// ==============================================================================
// Front-End Macros
// ==============================================================================

// (Your standard defines)
#define MDMP_IGNORE -2
#define MDMP_SUM 0
#define MDMP_MAX 1
#define MDMP_MIN 2

#define MDMP_SEND(buf, count, sender, dest, tag) \
    __mdmp_marker_send((void*)(buf), count, MDMP_DEDUCE_TYPE(buf), (count) * sizeof(*(buf)), sender, dest, tag)

#define MDMP_RECV(buf, count, receiver, src, tag) \
    __mdmp_marker_recv((void*)(buf), count, MDMP_DEDUCE_TYPE(buf), (count) * sizeof(*(buf)), receiver, src, tag)

#define MDMP_REDUCE(in_buf, out_buf, count, root, op) \
    __mdmp_marker_reduce((void*)(in_buf), (void*)(out_buf), count, MDMP_DEDUCE_TYPE(in_buf), (count) * sizeof(*(in_buf)), root, op)
#define MDMP_GATHER(send_buf, send_count, recv_buf, root) \
    __mdmp_marker_gather((void*)(send_buf), send_count, (void*)(recv_buf), MDMP_DEDUCE_TYPE(send_buf), (send_count) * sizeof(*(send_buf)), root)

#define MDMP_COMM_INIT()        __mdmp_marker_init()
#define MDMP_COMM_FINAL()       __mdmp_marker_final()
#define MDMP_COMMREGION_BEGIN() __mdmp_marker_commregion_begin()
#define MDMP_COMMREGION_END()   __mdmp_marker_commregion_end()
#define MDMP_COMM_SYNC()        __mdmp_marker_sync()

#define MDMP_GET_SIZE()         __mdmp_marker_get_size()
#define MDMP_GET_RANK()         __mdmp_marker_get_rank()
#define MDMP_RANK               __mdmp_marker_get_rank() // Alias for cleaner maths
#define MDMP_WTIME()            __mdmp_marker_wtime()

#define MDMP_SET_DEBUG(enable)        mdmp_set_debug(enable)


#endif
