// mdmp_pragma_interface.h
#ifndef MDMP_PRAGMA_INTERFACE_H
#define MDMP_PRAGMA_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

// Pragma-based MDMP directives
#define MDMP_PRAGMA(x) _Pragma(#x)

// Communication patterns
#define MDMP_COMMREGION_BEGIN() \
    MDMP_PRAGMA(mdmp overlap begin)

#define MDMP_COMMREGION_END() \
    MDMP_PRAGMA(mdmp overlap end)

#define MDMP_COMM_SYNC() \
    MDMP_PRAGMA(mdmp sync)

#define MDMP_COMM_WAIT() \
    MDMP_PRAGMA(mdmp wait)

#define MDMP_COMM_RANK() \
    MDMP_PRAGMA(mdmp rank)

#define MDMP_COMM_SIZE() \
    MDMP_PRAGMA(mdmp size)

#define MDMP_COMM_SEND() \
    MDMP_PRAGMA(mdmp send)

#define MDMP_COMM_RECV() \
    MDMP_PRAGMA(mdmp recv)

// Collective operations
#define MDMP_BARRIER() \
    MDMP_PRAGMA(mdmp barrier)

#define MDMP_BROADCAST(data, size, root) \
    MDMP_PRAGMA(mdmp broadcast data size root)

#define MDMP_GATHER(sendbuf, recvbuf, count, datatype, root) \
    MDMP_PRAGMA(mdmp gather sendbuf recvbuf count datatype root)

#define MDMP_SCATTER(sendbuf, recvbuf, count, datatype, root) \
    MDMP_PRAGMA(mdmp scatter sendbuf recvbuf count datatype root)

// Optimization control
#define MDMP_OPT_ENABLE() \
    MDMP_PRAGMA(mdmp enable)

#define MDMP_OPT_DISABLE() \
    MDMP_PRAGMA(mdmp disable)

#define MDMP_OPT_LEVEL(level) \
    MDMP_PRAGMA(mdmp level level)

#ifdef __cplusplus
}
#endif

#endif

