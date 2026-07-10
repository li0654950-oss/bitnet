#ifndef CIM_IPC_H
#define CIM_IPC_H
/* IPC client: unix socket 转发 4 个 MMIO 回调到 Python HwCimSimulator server。
 * cim_ipc_init 连接 server + register_cim_hw_sim(IPC 回调) -> cim_stub HW_READY=1。
 * 之后 cim_stub 的 cim_load_forward / cim_preload_init / cim_launch 经 IPC 驱动仿真器。
 *
 * 协议 (小端, 请求-响应):
 *   shm_write: req [op=1 | off(8) | n(8) | data(n)]  -> resp [ack(1)]
 *   shm_read:  req [op=2 | off(8) | n(8)]            -> resp [data(n)]
 *   reg_write: req [op=3 | reg(8) | val(8)]          -> resp [ack(1)]
 *   reg_read:  req [op=4 | reg(8)]                   -> resp [val(4)]
 */
int cim_ipc_init(const char* socket_path);

#endif
