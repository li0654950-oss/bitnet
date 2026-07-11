#ifndef CIM_SHM_H
#define CIM_SHM_H
#include <stdint.h>
#include <stddef.h>
#include "hw_config.h"   /* SHARED_SIZE (CIM 共享缓存 1MB, 集中定义) */

/* POSIX 共享内存: C (cim_sim) 与 Python (cim_sim_server.py SharedMemory) 共享 1MB mmap,
 * 承载 CIM 共享缓存 (SharedCache.data, hw_simulator.py)。shm_* 零拷贝零往返 (替代 socket)。
 *
 * 名字约定: C shm_open("/cim_cache") == Python SharedMemory(name="cim_cache") (内部加 /)。
 * server 创建 (cim_shm_create, 先启动), cim_sim 打开 (cim_shm_open, 后启动)。
 *
 * 同步: shm_* 数据传输走共享内存 (无 socket), reg_* 控制走 socket 作同步点
 *   (C 写 shm -> reg_write DOORBELL socket -> Python 读; Python 写 PSUM -> irq DONE socket -> C 读)。
 */

#define CIM_SHM_NAME "/cim_cache"
#define CIM_SHM_SIZE SHARED_SIZE   /* 1MB, = hw_config.SHARED_SIZE (CIM 共享缓存) */

/* 创建共享内存 (server 用): shm_open(O_CREAT) + ftruncate + mmap -> 指针 */
void* cim_shm_create(const char* name, size_t size);

/* 打开已有共享内存 (cim_sim 用): shm_open + mmap -> 指针 */
void* cim_shm_open(const char* name, size_t size);

/* munmap (cim_sim 退出) */
void cim_shm_unmap(void* ptr, size_t size);

/* shm_unlink 删除共享内存 (server 退出, 清理) */
void cim_shm_unlink(const char* name);

#endif /* CIM_SHM_H */
