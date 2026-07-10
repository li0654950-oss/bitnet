#!/bin/bash
# AOT 系统仿真一键启动 (终端用): cim_sim_server.py (Python 仿真器) + cim_sim (AOT 可执行文件)
#
# 用法:
#   ./cim_compiler/lowering/aot/run_aot.sh                            # 默认 prompt=ROMEO: n=60
#   ./cim_compiler/lowering/aot/run_aot.sh --prompt "ROMEO:" --n 20   # 自定义参数
#
# 流程: 构建 cim_sim (若缺) -> 启动 server (nohup 后台) -> 跑 cim_sim -> kill server
cd "$(dirname "$0")/../../.."   # repo root

PY=/home/li/anaconda3/envs/nanogpt-gpu/bin/python
SOCK=/tmp/cim_sim.sock
AOT=cim_compiler/lowering/aot/cim_sim
SERVER=cim_compiler/lowering/aot/cim_sim_server.py

# 1. 构建 cim_sim (若不存在)
if [ ! -x "$AOT" ]; then
    echo "[run_aot] 构建 cim_sim (make)..."
    make -C cim_compiler/lowering/aot || exit 1
fi

# 2. 启动 server (nohup 后台, 不阻塞 script 退出)
pkill -f cim_sim_server 2>/dev/null
rm -f "$SOCK"
echo "[run_aot] 启动 CIM 仿真器 server..."
nohup "$PY" "$SERVER" --socket "$SOCK" >/tmp/cim_sim_server.log 2>&1 &
SERVER_PID=$!

# 3. 等 server 监听就绪 (import torch_mlir 慢, poll socket 文件)
READY=0
for i in $(seq 1 40); do
    if [ -S "$SOCK" ]; then READY=1; break; fi
    sleep 0.5
done
if [ "$READY" = "0" ]; then
    echo "[run_aot] server 启动超时, 日志:"; tail -5 /tmp/cim_sim_server.log
    kill -9 $SERVER_PID 2>/dev/null; exit 1
fi

# 4. 跑 cim_sim (透传参数, 默认 prompt=ROMEO: n=60)
if [ $# -eq 0 ]; then
    "$AOT" --socket "$SOCK" --prompt "ROMEO:" --n 60
else
    "$AOT" --socket "$SOCK" "$@"
fi
CIM_RC=$?

# 5. 关 server (循环 accept 持久, 须显式 kill; sleep 等 server 打印统计)
sleep 1
kill -9 $SERVER_PID 2>/dev/null
rm -f "$SOCK"
echo "[run_aot] server 统计 (tail):"; tail -2 /tmp/cim_sim_server.log 2>/dev/null
exit $CIM_RC
