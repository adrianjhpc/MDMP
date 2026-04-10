#include "mdmp_interface.h"

void do_mdmp_send(double* buf, int count, int actor, int peer, int tag) {
    MDMP_SEND(buf, count, actor, peer, tag);
}

void do_mdmp_recv(double* buf, int count, int actor, int peer, int tag) {
    MDMP_RECV(buf, count, actor, peer, tag);
}
