#!/usr/bin/env python3
"""S0 自测: verify 错误注入 + pass round-trip。

构造 1 个好 IR + 3 个坏 IR (重复 dest_id / accum 链错 / PAGE 并行冲突),
验证 verify 能精准拦截。再验证 canonicalize/cse round-trip 不改好 IR 语义。

用法:
  nanogpt-gpu python cim_compiler/cimres/passes/_s0_selftest.py
"""
import sys

from cim_compiler.cimres.passes.common import parse_cimres
from cim_compiler.cimres.verify import verify
from cim_compiler.cimres.passes.canonicalize import canonicalize
from cim_compiler.cimres.passes.cse import cse

# 2x2 tile (N=K=128), dest_id = n_blk*2 + k_blk: (n0,k0)=0 (n1,k0)=2 (n0,k1)=1 (n1,k1)=3
# k_blk=0: dest 0,2 accum=false psum 3072,3073; k_blk=1: dest 1,3 accum=true psum 3072,3073
GOOD = '''module {
  "cimres.tile_group"() {K = 128 : i32, N = 128 : i32, bitlinear_name = "t"} : () -> ()
  "cimres.preload_weight"() {b_page_start = 0 : i32, bitlinear_name = "t", dest_id = 0 : i32} : () -> ()
  "cimres.preload_weight"() {b_page_start = 4 : i32, bitlinear_name = "t", dest_id = 1 : i32} : () -> ()
  "cimres.preload_weight"() {b_page_start = 8 : i32, bitlinear_name = "t", dest_id = 2 : i32} : () -> ()
  "cimres.preload_weight"() {b_page_start = 12 : i32, bitlinear_name = "t", dest_id = 3 : i32} : () -> ()
  func.func @cim_t(%arg0: tensor<128xi8>) -> tensor<64xi32> {
    %0 = "cimres.macro_matmul"(%arg0) {a_page = 16 : i32, accum = false, bitlinear_name = "t", dest_id = 0 : i32, k_blk = 0 : i32, n_blk = 0 : i32, psum_page = 3072 : i32} : (tensor<128xi8>) -> tensor<64xi32>
    %1 = "cimres.macro_matmul"(%arg0) {a_page = 16 : i32, accum = false, bitlinear_name = "t", dest_id = 2 : i32, k_blk = 0 : i32, n_blk = 1 : i32, psum_page = 3073 : i32} : (tensor<128xi8>) -> tensor<64xi32>
    %2 = "cimres.macro_matmul"(%arg0) {a_page = 17 : i32, accum = true, bitlinear_name = "t", dest_id = 1 : i32, k_blk = 1 : i32, n_blk = 0 : i32, psum_page = 3072 : i32} : (tensor<128xi8>) -> tensor<64xi32>
    %3 = "cimres.macro_matmul"(%arg0) {a_page = 17 : i32, accum = true, bitlinear_name = "t", dest_id = 3 : i32, k_blk = 1 : i32, n_blk = 1 : i32, psum_page = 3073 : i32} : (tensor<128xi8>) -> tensor<64xi32>
    "cimres.sync_halt"() : () -> ()
    return %3 : tensor<64xi32>
  }
}'''

# 坏 1: 重复 dest_id preload (多一个 dest_id=0)
DUP_DEST = GOOD.replace(
    '  "cimres.tile_group"() {K = 128 : i32, N = 128 : i32, bitlinear_name = "t"} : () -> ()\n',
    '  "cimres.tile_group"() {K = 128 : i32, N = 128 : i32, bitlinear_name = "t"} : () -> ()\n'
    '  "cimres.preload_weight"() {b_page_start = 16 : i32, bitlinear_name = "t", dest_id = 0 : i32} : () -> ()\n')

# 坏 2: accum 链错 (n0 k0 应 false 却 true)
ACCUM_CHAIN = GOOD.replace(
    'dest_id = 0 : i32, k_blk = 0 : i32, n_blk = 0 : i32, psum_page = 3072 : i32} : (tensor<128xi8>) -> tensor<64xi32>',
    'dest_id = 0 : i32, k_blk = 0 : i32, n_blk = 0 : i32, psum_page = 3072 : i32} : (tensor<128xi8>) -> tensor<64xi32>',
    1).replace(
    'a_page = 16 : i32, accum = false, bitlinear_name = "t", dest_id = 0 : i32',
    'a_page = 16 : i32, accum = true, bitlinear_name = "t", dest_id = 0 : i32', 1)

# 坏 3: PAGE 并行冲突 (k0 的 n0/n1 都用 psum 3072)
PAGE_CONFLICT = GOOD.replace(
    'dest_id = 2 : i32, k_blk = 0 : i32, n_blk = 1 : i32, psum_page = 3073 : i32',
    'dest_id = 2 : i32, k_blk = 0 : i32, n_blk = 1 : i32, psum_page = 3072 : i32', 1)


def run():
    ok = True
    # ---- 错误注入: 好 IR 通过, 坏 IR 精准拦截 ----
    cases = [
        ("GOOD",          GOOD,          None,            "应通过"),
        ("DUP_DEST",      DUP_DEST,      "dest_id 重复",  "重复 dest_id preload"),
        ("ACCUM_CHAIN",   ACCUM_CHAIN,   "accum",         "accum 链错 (k0 应 false)"),
        ("PAGE_CONFLICT", PAGE_CONFLICT, "并行冲突",      "k0 并行写同 psum_page"),
    ]
    print("=== verify 错误注入 ===")
    for name, src, expect_kw, desc in cases:
        mod, _ = parse_cimres(src)
        issues = verify(mod)
        if expect_kw is None:
            passed = len(issues) == 0
            print(f"  [{name}] {'PASS ✓' if passed else 'FAIL ✗'} ({desc}): issues={len(issues)}")
            if not passed:
                for s in issues:
                    print(f"      {s}")
                ok = False
        else:
            hit = any(expect_kw in s for s in issues)
            print(f"  [{name}] {'PASS ✓' if hit else 'FAIL ✗'} ({desc}): expect '{expect_kw}'")
            if not hit:
                print(f"      实际 issues: {issues}")
                ok = False

    # ---- pass round-trip: 好 IR 经 canon+cse 后仍通过 verify 且不坏结构 ----
    print("=== pass round-trip ===")
    mod, _ = parse_cimres(GOOD)
    before = str(mod)
    nc = canonicalize(mod)
    ncs = cse(mod)
    after = str(mod)
    issues = verify(mod)
    rt_ok = (len(issues) == 0 and nc == 0 and ncs == 0)
    # round-trip 不改好 IR (规整 IR 无冗余可消)
    unchanged = before == after
    print(f"  canon 消除 {nc}, cse 消除 {ncs}, verify 后 issues={len(issues)}, IR 不变={'是' if unchanged else '否'}")
    if not (rt_ok and unchanged):
        ok = False
        print(f"  FAIL ✗: 预期 canon=0 cse=0 issues=0 IR 不变")

    print(f"\n[S0 self-test] {'PASS ✓' if ok else 'FAIL ✗'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    run()
