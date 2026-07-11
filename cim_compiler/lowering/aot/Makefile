# AOT 系统仿真: 模型 .o + cim_stub + cim_runtime + cim_ipc + cim_shm + cim_main -> cim_sim
#
# 用法 (从 repo root 或 aot/ 均可):
#   make            # 全量构建 cim_sim + model_config.bin
#   make run        # 自动启动 server + 跑 cim_sim (n=3)
#   make clean      # 清中间产物 (保留 .o)
#
# 阶段2: IPC 仿真器 (cim_sim 经 unix socket 驱动 Python HwCimSimulator)
# cim_main.c 固定通用宿主: 超参运行时从 model_config.bin 读 (gen_config.py 从 .pt2 提取),
#   forward 参数个数随 n_layer 变由 libffi 运行时变参调用 (ffi_call) 解决。
CC = gcc
CFLAGS = -O2 -fPIC -Wall
PY = /home/li/anaconda3/envs/nanogpt-gpu/bin/python

HERE = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT = $(abspath $(HERE)/../../..)
STUB_C = $(ROOT)/cim_compiler/lowering/cim_stub.c

OBJ = $(ROOT)/checkpoints/bitnet_ternary.o
STUB_O = cim_stub.o
RUNTIME_O = cim_runtime.o
IPC_O = cim_ipc.o
SHM_O = cim_shm.o
MAIN_O = cim_main.o
RUNTIME_SO = cim_runtime.so
SIM = cim_sim
SOCK = /tmp/cim_sim.sock
CONFIG = $(ROOT)/cim_compiler/cimres/checkpoints/model_config.bin

all: $(SIM) $(CONFIG)

# 1. cim_runtime.so (C 版 consume ciface, to_object.py dump .o 要求 symbol resolved)
$(RUNTIME_SO): cim_runtime.c cim_runtime.h
	$(CC) -shared $(CFLAGS) -o $@ cim_runtime.c

# 2. to_object.py dump .o + objcopy 改名 main->forward_entry (避免与 cim_main 的 main 冲突)
$(OBJ): $(ROOT)/checkpoints/bitnet_ternary_placeholder.mlir $(STUB_C) $(RUNTIME_SO)
	cd $(ROOT) && $(PY) cim_compiler/lowering/to_object.py \
		--in checkpoints/bitnet_ternary_placeholder.mlir \
		--so cim_compiler/lowering/cim_stub.so \
		--runtime-so cim_compiler/lowering/aot/$(RUNTIME_SO) \
		--out checkpoints/bitnet_ternary.o
	objcopy --redefine-sym main=forward_entry $(OBJ)

# 3. cim_stub.o (cim_launch + register_cim_hw_sim + cim_load_forward + cim_preload_init)
$(STUB_O): $(STUB_C)
	$(CC) $(CFLAGS) -c -o $@ $(STUB_C)

# 4. cim_runtime.o (consume ciface + unranked memref 工具)
$(RUNTIME_O): cim_runtime.c cim_runtime.h
	$(CC) $(CFLAGS) -c -o $@ cim_runtime.c

# 5. cim_ipc.o (IPC client: shm 共享内存 + reg socket)
$(IPC_O): cim_ipc.c cim_ipc.h cim_shm.h
	$(CC) $(CFLAGS) -c -o $@ cim_ipc.c

# 5b. cim_shm.o (POSIX 共享内存抽象, shm_open/mmap)
$(SHM_O): cim_shm.c cim_shm.h
	$(CC) $(CFLAGS) -c -o $@ cim_shm.c

# 6. cim_main.o (固定通用入口: load_config + ipc_init + load_forward + preload_init + generate 循环 + libffi 调 forward)
$(MAIN_O): cim_main.c cim_runtime.h cim_ipc.h model_config.h
	$(CC) $(CFLAGS) -c -o $@ cim_main.c

# 7. 链接 cim_sim (-lrt: shm_open, -lffi: 运行时变参调 forward)
$(SIM): $(OBJ) $(STUB_O) $(RUNTIME_O) $(IPC_O) $(SHM_O) $(MAIN_O)
	$(CC) -o $@ $(OBJ) $(STUB_O) $(RUNTIME_O) $(IPC_O) $(SHM_O) $(MAIN_O) -lm -lrt -lffi

# 8. gen_config: 从 .pt2 提取 model_config.bin (cim_main 运行时读, 规模参数化)
$(CONFIG): $(ROOT)/checkpoints/bitnet_ternary.pt2 $(HERE)/gen_config.py
	cd $(ROOT) && $(PY) cim_compiler/lowering/aot/gen_config.py

# 自动启动 server + 跑 cim_sim (从 repo root, forward/preload/config 路径相对 repo root)
run: $(SIM) $(CONFIG)
	cd $(ROOT) && $(PY) cim_compiler/lowering/aot/cim_sim_server.py --socket $(SOCK) & \
	sleep 1 && cd $(ROOT) && ./cim_compiler/lowering/aot/$(SIM) --socket $(SOCK) --n 3; \
	wait

clean:
	rm -f $(STUB_O) $(RUNTIME_O) $(IPC_O) $(SHM_O) $(MAIN_O) $(RUNTIME_SO) $(SIM)
