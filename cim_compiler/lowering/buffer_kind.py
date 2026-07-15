"""BUFFER target -> kind 分类 (cim_jit.build_inputs + gen_config 共用单一来源)。

ExportedProgram.graph_signature.input_specs 的 BUFFER target 是 PyTorch 参数名
(如 layers.0.attn.q_proj.w_packed / layers.0.attn.inv_freq / lm_head.w_packed)。
ExportedProgram 不提供 "这是 inv_freq" 之类语义元数据, 只能靠命名约定分类
(B 类: 无结构化替代, 本模块消除 cim_jit / gen_config 的重复 if-elif)。
"""
KIND_INVFREQ, KIND_CAUSAL_MASK, KIND_W_PACKED, KIND_LMHEAD = 0, 1, 2, 3
KIND_NAME = {0: "inv_freq", 1: "causal_mask", 2: "w_packed", 3: "lmhead_w"}


def classify_buffer(target):
    """按 target 命名约定分类 buffer kind。

    与原 gen_config.classify / cim_jit build_inputs if-elif 等价:
      inv_freq         -> KIND_INVFREQ      (RoPE 频率, float32)
      causal_mask      -> KIND_CAUSAL_MASK  (bool)
      lm_head.w_packed -> KIND_LMHEAD       (反推 vocab; cim_jit 仍按 w_packed 转 int8)
      *.w_packed       -> KIND_W_PACKED     (ternary 权重, uint8 -> i8 view)
    """
    if "inv_freq" in target:
        return KIND_INVFREQ
    if "causal_mask" in target:
        return KIND_CAUSAL_MASK
    if "lm_head" in target and target.endswith("w_packed"):
        return KIND_LMHEAD
    if target.endswith("w_packed"):
        return KIND_W_PACKED
    raise ValueError(f"未知 buffer target: {target}")
