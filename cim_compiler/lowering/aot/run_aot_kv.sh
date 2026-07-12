#!/bin/bash
# AOT 增量 KV 仿真一键启动: cim_sim_server.py + cim_sim_kv (--kv, O(n²)->O(n))
#
# 用法:
#   ./cim_compiler/lowering/aot/run_aot_kv.sh                            # 默认 prompt=ROMEO: n=60
#   ./cim_compiler/lowering/aot/run_aot_kv.sh --prompt "ROMEO:" --n 20   # 自定义参数
cd "$(dirname "$0")/../../.."   # repo root

PY=/home/li/anaconda3/envs/nanogpt-gpu/bin/python
SOCK=/tmp/cim_sim.sock
AOT=cim_compiler/lowering/aot/cim_sim_kv
SERVER=cim_compiler/lowering/aot/cim_sim_server.py

# 1. 构建 cim_sim_kv (若不存在)
if [ ! -x "$AOT" ]; then
    echo "[run_aot_kv] 构建 cim_sim_kv (make)..."
    make -C cim_compiler/lowering/aot cim_sim_kv || exit 1
fi

# 2. 启动 server (nohup 后台)
pkill -f cim_sim_server 2>/dev/null
rm -f "$SOCK"
echo "[run_aot_kv] 启动 CIM 仿真器 server..."
nohup "$PY" "$SERVER" --socket "$SOCK" >/tmp/cim_sim_server.log 2>&1 &
SERVER_PID=$!

# 3. 等 server 监听就绪
READY=0
for i in $(seq 1 40); do
    if [ -S "$SOCK" ]; then READY=1; break; fi
    sleep 0.5
done
if [ "$READY" = "0" ]; then
    echo "[run_aot_kv] server 启动超时, 日志:"; tail -5 /tmp/cim_sim_server.log
    kill -9 $SERVER_PID 2>/dev/null; exit 1
fi

# 4. 跑 cim_sim_kv --kv (透传参数, 默认 prompt=ROMEO: n=60)
if [ $# -eq 0 ]; then
    "$AOT" --socket "$SOCK" --kv --prompt "ROMEO:" --n 60
else
    "$AOT" --socket "$SOCK" --kv "$@"
fi
CIM_RC=$?

# 5. 关 server
sleep 1
kill -9 $SERVER_PID 2>/dev/null
rm -f "$SOCK"
echo "[run_aot_kv] server 统计 (tail):"; tail -2 /tmp/cim_sim_server.log 2>/dev/null
exit $CIM_RC
