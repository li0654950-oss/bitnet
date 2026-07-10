/* IPC client: 共享内存 (shm 数据) + unix socket (reg 控制)。
 *
 * shm_* (CIM 共享缓存数据传输, 大块高频): 直接 memcpy 共享内存 (零拷贝零往返)。
 * reg_* (门铃/IRQ/INT_CLEAR 控制, 少量): 走 socket (作同步点, 保证 C 写/Python 读可见性)。
 *
 * cim_stub.c 不改: register_cim_hw_sim 注册 4 回调 (shm 走共享内存, reg 走 socket)。
 * cim_sim_server.py 创建共享内存 + sim.cache.data backed by it, socket 只处理 reg_*。
 */
#include "cim_ipc.h"
#include "cim_shm.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

/* cim_stub.c 的符号 (链接 cim_stub.o) */
typedef void (*shm_write_cb_t)(long, const void*, long);
typedef void (*shm_read_cb_t)(long, void*, long);
typedef void (*reg_write_cb_t)(long, long);
typedef int32_t (*reg_read_cb_t)(long);
extern void register_cim_hw_sim(shm_write_cb_t, shm_read_cb_t, reg_write_cb_t, reg_read_cb_t);

static int g_sock = -1;
static void* g_shm = NULL;   /* 共享内存指针 (CIM 共享缓存, 1MB, C/Python 共享) */

static int send_all(const void* buf, long n) {
    long sent = 0;
    while (sent < n) {
        long r = (long)send(g_sock, (const char*)buf + sent, (size_t)(n - sent), 0);
        if (r <= 0) return -1;
        sent += r;
    }
    return 0;
}
static int recv_all(void* buf, long n) {
    long got = 0;
    while (got < n) {
        long r = (long)recv(g_sock, (char*)buf + got, (size_t)(n - got), 0);
        if (r <= 0) return -1;
        got += r;
    }
    return 0;
}

/* shm_* 直接读写共享内存 (零拷贝零往返, 替代 socket)。
 * 同步靠 reg_* socket: C 写 shm -> reg_write(DOORBELL) socket -> Python 读;
 * Python 写 PSUM -> irq DONE socket -> C shm_read。socket 往返保证可见性。 */
static void ipc_shm_write(long off, const void* data, long n) {
    if (g_shm) memcpy((char*)g_shm + off, data, n);
}
static void ipc_shm_read(long off, void* buf, long n) {
    if (g_shm) memcpy(buf, (char*)g_shm + off, n);
}

/* reg_* 走 socket (控制信号, op=3 reg_write / op=4 reg_read) */
static void ipc_reg_write(long reg, long val) {
    char op = 3;
    send_all(&op, 1); send_all(&reg, 8); send_all(&val, 8);
    char ack; recv_all(&ack, 1);
}
static int32_t ipc_reg_read(long reg) {
    char op = 4;
    send_all(&op, 1); send_all(&reg, 8);
    int32_t val; recv_all(&val, 4);
    return val;
}

int cim_ipc_init(const char* socket_path) {
    /* 1. 连 socket (reg_* 控制通道) */
    g_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (g_sock < 0) { perror("socket"); return -1; }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
    if (connect(g_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("cim_ipc connect"); return -1;
    }
    /* 2. 打开共享内存 (shm_* 数据通道, server 已创建) */
    g_shm = cim_shm_open(CIM_SHM_NAME, CIM_SHM_SIZE);
    if (!g_shm) {
        fprintf(stderr, "[cim_ipc] 共享内存 '%s' 打开失败 (server 未启动?)\n", CIM_SHM_NAME);
        return -1;
    }
    /* 3. 注册 4 回调 (shm 走共享内存, reg 走 socket) */
    register_cim_hw_sim(ipc_shm_write, ipc_shm_read, ipc_reg_write, ipc_reg_read);
    printf("[cim_ipc] server + 共享内存 '%s' (%dB) 就绪 (shm 共享内存 / reg socket)\n",
           CIM_SHM_NAME, CIM_SHM_SIZE);
    return 0;
}
