/* POSIX 共享内存实现: shm_open + ftruncate + mmap (Linux /dev/shm)。
 * C 与 Python (multiprocessing.shared_memory.SharedMemory) 共享 1MB, 承载 CIM 共享缓存。
 */
#include "cim_shm.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>

static void* mmap_shm(int fd, size_t size) {
    void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap"); return NULL; }
    return p;
}

void* cim_shm_open(const char* name, size_t size) {
    /* 仅打开 (server 已创建); 不 O_CREAT (cim_sim 后启动) */
    int fd = shm_open(name, O_RDWR, 0666);
    if (fd < 0) { perror("shm_open open (server 未启动?)"); return NULL; }
    return mmap_shm(fd, size);
}
