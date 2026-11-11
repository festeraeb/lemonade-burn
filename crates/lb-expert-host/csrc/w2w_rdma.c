/* csrc/w2w_rdma.c - the libibverbs calls the expert-host RDMA transport needs, as plain functions.
 *
 * ibv_post_send(), ibv_post_recv() and ibv_poll_cq() are inline functions in <infiniband/verbs.h> that go
 * through the provider's ops table; Rust cannot call them without copying that struct's layout.  This file is
 * the one place that includes the header.  It follows the shape of our Nauti-Virtualizer nauti-fabric shim
 * (one reliable-connected queue pair per handle, the peer's numbers carried over a control link) and adds
 * what a request/reply needs: SEND/RECV with a ring of posted receive buffers, and a wait that spins, sleeps
 * on the completion channel, or spins then sleeps.
 */
#include <infiniband/verbs.h>

#include <errno.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define W2W_SEND_ID UINT64_MAX
#define W2W_RING 64

struct w2w_rdma {
    struct ibv_context* ctx;
    struct ibv_pd* pd;
    struct ibv_comp_channel* ch;
    struct ibv_cq* cq;
    struct ibv_qp* qp;
    int port;
    int gid_index;
    uint32_t psn;
    uint16_t lid;
    int mtu; /* enum ibv_mtu of the port */
    union ibv_gid gid;
    size_t max_msg;
    int n_recv;
    uint8_t* send_buf;
    struct ibv_mr* send_mr;
    uint8_t* recv_buf; /* n_recv * max_msg */
    struct ibv_mr* recv_mr;
    /* receives completed but not yet handed to the caller */
    int ready_idx[W2W_RING];
    uint32_t ready_len[W2W_RING];
    int ready_head, ready_n;
};

static void say(char* err, size_t errlen, const char* what, int code) {
    if (!err || errlen == 0) return;
    if (code) snprintf(err, errlen, "%s: %s", what, strerror(code));
    else snprintf(err, errlen, "%s", what);
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
}

void w2w_rdma_close(struct w2w_rdma* h) {
    if (!h) return;
    if (h->qp) ibv_destroy_qp(h->qp);
    if (h->cq) ibv_destroy_cq(h->cq);
    if (h->ch) ibv_destroy_comp_channel(h->ch);
    if (h->send_mr) ibv_dereg_mr(h->send_mr);
    if (h->recv_mr) ibv_dereg_mr(h->recv_mr);
    if (h->pd) ibv_dealloc_pd(h->pd);
    if (h->ctx) ibv_close_device(h->ctx);
    free(h->send_buf);
    free(h->recv_buf);
    free(h);
}

struct w2w_rdma* w2w_rdma_open(const char* dev, int port, int gid_index, size_t max_msg, int n_recv, char* err,
                               size_t errlen) {
    if (n_recv < 1 || n_recv > W2W_RING || max_msg < 64) {
        say(err, errlen, "bad buffer geometry", 0);
        return NULL;
    }
    int n = 0;
    struct ibv_device** list = ibv_get_device_list(&n);
    if (!list) {
        say(err, errlen, "ibv_get_device_list", errno);
        return NULL;
    }
    struct ibv_device* found = NULL;
    for (int i = 0; i < n; ++i)
        if (strcmp(ibv_get_device_name(list[i]), dev) == 0) found = list[i];
    if (!found) {
        snprintf(err, errlen, "no RDMA device %s (%d seen)", dev, n);
        ibv_free_device_list(list);
        return NULL;
    }
    struct w2w_rdma* h = calloc(1, sizeof(*h));
    if (!h) {
        ibv_free_device_list(list);
        say(err, errlen, "calloc", errno);
        return NULL;
    }
    h->port = port;
    h->gid_index = gid_index;
    h->max_msg = max_msg;
    h->n_recv = n_recv;
    h->ctx = ibv_open_device(found);
    ibv_free_device_list(list);
    if (!h->ctx) {
        say(err, errlen, "ibv_open_device", errno);
        free(h);
        return NULL;
    }
    struct ibv_port_attr pa;
    if (ibv_query_port(h->ctx, (uint8_t) port, &pa)) {
        say(err, errlen, "ibv_query_port", errno);
        goto fail;
    }
    if (pa.state != IBV_PORT_ACTIVE) {
        snprintf(err, errlen, "%s port %d is not active (state %d)", dev, port, (int) pa.state);
        goto fail;
    }
    h->lid = pa.lid;
    h->mtu = pa.active_mtu;
    if (ibv_query_gid(h->ctx, (uint8_t) port, gid_index, &h->gid)) {
        say(err, errlen, "ibv_query_gid", errno);
        goto fail;
    }
    h->pd = ibv_alloc_pd(h->ctx);
    if (!h->pd) {
        say(err, errlen, "ibv_alloc_pd", errno);
        goto fail;
    }
    if (posix_memalign((void**) &h->send_buf, 4096, max_msg) || posix_memalign((void**) &h->recv_buf, 4096, max_msg * (size_t) n_recv)) {
        say(err, errlen, "posix_memalign", ENOMEM);
        goto fail;
    }
    memset(h->send_buf, 0, max_msg);
    memset(h->recv_buf, 0, max_msg * (size_t) n_recv);
    h->send_mr = ibv_reg_mr(h->pd, h->send_buf, max_msg, IBV_ACCESS_LOCAL_WRITE);
    h->recv_mr = ibv_reg_mr(h->pd, h->recv_buf, max_msg * (size_t) n_recv, IBV_ACCESS_LOCAL_WRITE);
    if (!h->send_mr || !h->recv_mr) {
        say(err, errlen, "ibv_reg_mr", errno);
        goto fail;
    }
    h->ch = ibv_create_comp_channel(h->ctx);
    if (!h->ch) {
        say(err, errlen, "ibv_create_comp_channel", errno);
        goto fail;
    }
    h->cq = ibv_create_cq(h->ctx, n_recv + 16, NULL, h->ch, 0);
    if (!h->cq) {
        say(err, errlen, "ibv_create_cq", errno);
        goto fail;
    }
    struct ibv_qp_init_attr qa;
    memset(&qa, 0, sizeof(qa));
    qa.send_cq = h->cq;
    qa.recv_cq = h->cq;
    qa.cap.max_send_wr = 8;
    qa.cap.max_recv_wr = (uint32_t) n_recv;
    qa.cap.max_send_sge = 1;
    qa.cap.max_recv_sge = 1;
    qa.qp_type = IBV_QPT_RC;
    h->qp = ibv_create_qp(h->pd, &qa);
    if (!h->qp) {
        say(err, errlen, "ibv_create_qp", errno);
        goto fail;
    }
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = (uint8_t) port;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE;
    int rc = ibv_modify_qp(h->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (rc) {
        say(err, errlen, "ibv_modify_qp INIT", rc);
        goto fail;
    }
    srand48((long) time(NULL) ^ (long) h->qp->qp_num);
    h->psn = (uint32_t) lrand48() & 0xffffff;
    return h;
fail:
    w2w_rdma_close(h);
    return NULL;
}

void w2w_rdma_local(struct w2w_rdma* h, uint32_t* qpn, uint32_t* psn, uint8_t gid[16], uint16_t* lid, int* mtu) {
    *qpn = h->qp->qp_num;
    *psn = h->psn;
    memcpy(gid, h->gid.raw, 16);
    *lid = h->lid;
    *mtu = h->mtu;
}

static int post_recv(struct w2w_rdma* h, int idx) {
    struct ibv_sge sge = {.addr = (uintptr_t) (h->recv_buf + (size_t) idx * h->max_msg), .length = (uint32_t) h->max_msg,
                          .lkey = h->recv_mr->lkey};
    struct ibv_recv_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t) idx;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    return ibv_post_recv(h->qp, &wr, &bad);
}

/* Posts every receive buffer, then connects this queue pair to the peer's and brings it to RTS. */
int w2w_rdma_connect(struct w2w_rdma* h, uint32_t qpn, uint32_t psn, const uint8_t gid[16], uint16_t lid, int peer_mtu,
                     char* err, size_t errlen) {
    for (int i = 0; i < h->n_recv; ++i) {
        int rc = post_recv(h, i);
        if (rc) {
            say(err, errlen, "ibv_post_recv", rc);
            return -1;
        }
    }
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = (enum ibv_mtu) (peer_mtu < h->mtu ? peer_mtu : h->mtu);
    a.dest_qp_num = qpn;
    a.rq_psn = psn;
    a.max_dest_rd_atomic = 1;
    a.min_rnr_timer = 1; /* 0.01 ms: a receiver that is briefly out of buffers is retried quickly */
    a.ah_attr.is_global = 1;
    memcpy(a.ah_attr.grh.dgid.raw, gid, 16);
    a.ah_attr.grh.sgid_index = (uint8_t) h->gid_index;
    a.ah_attr.grh.hop_limit = 64;
    a.ah_attr.dlid = lid;
    a.ah_attr.port_num = (uint8_t) h->port;
    int rc = ibv_modify_qp(h->qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                      IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc) {
        say(err, errlen, "ibv_modify_qp RTR", rc);
        return -1;
    }
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS;
    a.timeout = 14;
    a.retry_cnt = 7;
    a.rnr_retry = 7; /* 7 = retry for ever: the receive ring is the flow control */
    a.sq_psn = h->psn;
    a.max_rd_atomic = 1;
    rc = ibv_modify_qp(h->qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                  IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc) {
        say(err, errlen, "ibv_modify_qp RTS", rc);
        return -1;
    }
    return 0;
}

/* Files one completion.  Returns 1 if it was the send's, 0 for a receive (queued), -1 on a failed one. */
static int file_wc(struct w2w_rdma* h, const struct ibv_wc* wc, char* err, size_t errlen) {
    if (wc->status != IBV_WC_SUCCESS) {
        snprintf(err, errlen, "%s: %s (%d)", wc->wr_id == W2W_SEND_ID ? "send" : "receive", ibv_wc_status_str(wc->status),
                 (int) wc->status);
        return -1;
    }
    if (wc->wr_id == W2W_SEND_ID) return 1;
    if (h->ready_n == W2W_RING) {
        say(err, errlen, "receive ring overflow", 0);
        return -1;
    }
    int at = (h->ready_head + h->ready_n) % W2W_RING;
    h->ready_idx[at] = (int) wc->wr_id;
    h->ready_len[at] = wc->byte_len;
    h->ready_n++;
    return 0;
}

/* Sends len bytes (one message, at most max_msg) and waits for its completion, so the send buffer is free
 * again on return.  Receives that complete meanwhile are queued for w2w_rdma_recv. */
int w2w_rdma_send(struct w2w_rdma* h, const void* data, size_t len, int timeout_ms, char* err, size_t errlen) {
    if (len > h->max_msg) {
        say(err, errlen, "message larger than the registered buffer", 0);
        return -1;
    }
    memcpy(h->send_buf, data, len);
    struct ibv_sge sge = {.addr = (uintptr_t) h->send_buf, .length = (uint32_t) len, .lkey = h->send_mr->lkey};
    struct ibv_send_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = W2W_SEND_ID;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    int rc = ibv_post_send(h->qp, &wr, &bad);
    if (rc) {
        say(err, errlen, "ibv_post_send", rc);
        return -1;
    }
    const double until = now_s() + timeout_ms / 1000.0;
    struct ibv_wc wc;
    for (unsigned spins = 0;; ++spins) {
        int n = ibv_poll_cq(h->cq, 1, &wc);
        if (n < 0) {
            say(err, errlen, "ibv_poll_cq", errno);
            return -1;
        }
        if (n == 1) {
            int k = file_wc(h, &wc, err, errlen);
            if (k < 0) return -1;
            if (k == 1) return 0;
        } else if ((spins & 0x3ff) == 0x3ff && now_s() > until) {
            snprintf(err, errlen, "send of %zu bytes: no completion within %d ms", len, timeout_ms);
            return -1;
        }
    }
}

/* Whether the peer has closed (or written to) the rendezvous socket: the pair is over. */
static int peer_gone(int watch_fd) {
    if (watch_fd < 0) return 0;
    struct pollfd p = {.fd = watch_fd, .events = POLLIN};
    return poll(&p, 1, 0) > 0;
}

/* Waits for one received message.  timeout_us < 0: wait for ever.  spin_us < 0: poll the completion queue
 * the whole time; 0: sleep on the completion channel at once; > 0: poll that long, then sleep.
 * watch_fd (or -1): the rendezvous socket; when it becomes readable the peer has gone.
 * Returns 0 with the buffer (valid until w2w_rdma_repost(idx)), 1 on a timeout, 2 when the peer has gone,
 * -1 on an error. */
int w2w_rdma_recv(struct w2w_rdma* h, const uint8_t** data, uint32_t* len, int* idx, int64_t timeout_us, int64_t spin_us,
                  int watch_fd, char* err, size_t errlen) {
    const double t0 = now_s();
    const double until = timeout_us < 0 ? 1e300 : t0 + (double) timeout_us * 1e-6;
    const double spin_end = spin_us < 0 ? 1e300 : t0 + (double) spin_us * 1e-6;
    double next_watch = t0 + 0.002;
    struct ibv_wc wc;
    for (unsigned spins = 0;; ++spins) {
        if (h->ready_n > 0) {
            *idx = h->ready_idx[h->ready_head];
            *len = h->ready_len[h->ready_head];
            *data = h->recv_buf + (size_t) *idx * h->max_msg;
            h->ready_head = (h->ready_head + 1) % W2W_RING;
            h->ready_n--;
            return 0;
        }
        int n = ibv_poll_cq(h->cq, 1, &wc);
        if (n < 0) {
            say(err, errlen, "ibv_poll_cq", errno);
            return -1;
        }
        if (n == 1) {
            if (file_wc(h, &wc, err, errlen) < 0) return -1;
            continue;
        }
        if (spin_us != 0 && (spins & 0x3f) != 0x3f) continue; /* read the clock every 64 empty polls */
        double now = now_s();
        if (now > until) return 1;
        if (now < spin_end) {
            /* spinning: look at the rendezvous socket every 2 ms (a syscall, so not every turn) */
            if (now > next_watch) {
                if (peer_gone(watch_fd)) return 2;
                next_watch = now + 0.002;
            }
            continue;
        }
        /* sleep: arm, look once more (a completion may have landed before the arm), then wait */
        if (ibv_req_notify_cq(h->cq, 0)) {
            say(err, errlen, "ibv_req_notify_cq", errno);
            return -1;
        }
        n = ibv_poll_cq(h->cq, 1, &wc);
        if (n < 0) {
            say(err, errlen, "ibv_poll_cq", errno);
            return -1;
        }
        if (n == 1) {
            if (file_wc(h, &wc, err, errlen) < 0) return -1;
            continue;
        }
        struct pollfd pfd[2] = {{.fd = h->ch->fd, .events = POLLIN}, {.fd = watch_fd, .events = POLLIN}};
        double left = until - now_s();
        int ms = timeout_us < 0 ? -1 : (left <= 0 ? 0 : (int) (left * 1000.0) + 1);
        int pr = poll(pfd, watch_fd < 0 ? 1 : 2, ms);
        if (pr < 0 && errno != EINTR) {
            say(err, errlen, "poll", errno);
            return -1;
        }
        if (pr > 0 && (pfd[0].revents & POLLIN)) {
            struct ibv_cq* ev_cq;
            void* ev_ctx;
            if (ibv_get_cq_event(h->ch, &ev_cq, &ev_ctx)) {
                say(err, errlen, "ibv_get_cq_event", errno);
                return -1;
            }
            ibv_ack_cq_events(ev_cq, 1);
        } else if (pr > 0) {
            /* only the rendezvous socket woke us: take what was already delivered, then report */
            n = ibv_poll_cq(h->cq, 1, &wc);
            if (n == 1) {
                if (file_wc(h, &wc, err, errlen) < 0) return -1;
                continue;
            }
            return 2;
        } else if (timeout_us >= 0 && now_s() > until) {
            /* one last look so a completion that raced the timeout is not lost */
            n = ibv_poll_cq(h->cq, 1, &wc);
            if (n == 1) {
                if (file_wc(h, &wc, err, errlen) < 0) return -1;
                continue;
            }
            return 1;
        }
    }
}

/* Gives a receive buffer back to the queue pair once the caller has copied the message out. */
int w2w_rdma_repost(struct w2w_rdma* h, int idx, char* err, size_t errlen) {
    int rc = post_recv(h, idx);
    if (rc) {
        say(err, errlen, "ibv_post_recv", rc);
        return -1;
    }
    return 0;
}
