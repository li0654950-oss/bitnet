"""cimres 优化 pass + 验证 (S0: 可优化可验证地基)。

pass 序列 (C1 -> C2 间):
  canonicalize -> cse -> verify(gate)

  - canonicalize: 规范化 (去重 preload / 合并 sync_halt / 死 preload 消除)
  - cse:          公共子表达式消除 (保守, macro_matmul 有副作用)
  - verify:       形式化验证 (dest_id 唯一 / accum 链 / PAGE 并行冲突 / a_page 一致)

verify 是安全网: 加 pass 后人工保证失效, 由 checker 兜底。
"""

