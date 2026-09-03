import argparse
from collections import Counter

import torch
import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}


@tilelang.jit(out_idx=[-1], workspace_idx=[-4, -3], pass_configs=pass_configs, target="ascendc")
def indexer(
    N2,
    G,
    D,
    TOP_K,
    VECTOR_BASEN,
    VECTOR_BASEG,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    MAX_S2=4096,
    input_dtype="float16",
    calc_dtype="float",
):
    B = T.symbolic("B")
    S1 = T.symbolic("S1")
    S2 = T.symbolic("S2")
    S1_TILES = S1 // BLOCK_M
    TASK_COUNT = B * N2 * S1_TILES
    READY_FLAG = 0
    FREE_FLAG = 2
    G_REDUCE_FLAG = 6
    SLOT_RELEASE_FLAG = 8
    L0C_FLAG = 10
    K_L1_READY_FLAG = 12
    Q_L1_READY_FLAG = 14
    OUTPUT_READY_FLAG = 16
    HISTORY_LOAD_FLAG = 18
    HISTORY_STORE_FLAG = 20
    K_L1_FREE_FLAG = 22
    Q_L1_FREE_FLAG = 24

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N2, G * D), input_dtype),
        KEY: T.Tensor((B, S2, N2, D), input_dtype),
        QK_SLOT: T.Tensor((TASK_COUNT, 2, BLOCK_M, G, BLOCK_N), calc_dtype),
        TOPK_HISTORY_WORKSPACE: T.Tensor((TASK_COUNT, BLOCK_M, 2 * TOP_K), calc_dtype),
        WEIGHTS: T.Tensor((B, S1, N2, G), calc_dtype),
        OUT: T.Tensor((B, N2, S1, TOP_K), "int32"),
    ):
        with T.Kernel(TASK_COUNT, threads=1, is_npu=True) as (cid):
            b_id = cid // (N2 * S1_TILES)
            n2_id = (cid // S1_TILES) % N2
            m_id = cid % S1_TILES

            with T.Scope("C"):
                q_l1 = T.alloc_L1((BLOCK_M, BLOCK_K), input_dtype)
                k_l1 = T.alloc_L1((BLOCK_N, BLOCK_K), input_dtype)
                c_l0 = T.alloc_L0C((BLOCK_M, BLOCK_N), calc_dtype)
                T.set_flag("FIX", "M", L0C_FLAG)
                T.set_flag("M", "MTE2", K_L1_FREE_FLAG)
                T.set_flag("M", "MTE2", Q_L1_FREE_FLAG)

                for n in T.serial(S2 // BLOCK_N):
                    slot = n % 2
                    T.wait_cross_flag(FREE_FLAG + slot)
                    T.wait_flag("M", "MTE2", K_L1_FREE_FLAG)
                    T.copy(KEY[b_id, n * BLOCK_N : (n + 1) * BLOCK_N, n2_id, 0:D], k_l1)
                    T.set_flag("MTE2", "M", K_L1_READY_FLAG)
                    T.wait_flag("MTE2", "M", K_L1_READY_FLAG)
                    for g in T.serial(G):
                        T.wait_flag("M", "MTE2", Q_L1_FREE_FLAG)
                        T.copy(
                            Query[
                                b_id,
                                m_id * BLOCK_M : (m_id + 1) * BLOCK_M,
                                n2_id,
                                g * D : (g + 1) * D,
                            ],
                            q_l1,
                        )
                        T.set_flag("MTE2", "M", Q_L1_READY_FLAG)
                        T.wait_flag("MTE2", "M", Q_L1_READY_FLAG)
                        T.wait_flag("FIX", "M", L0C_FLAG)
                        T.gemm_v0(q_l1, k_l1, c_l0, transpose_B=True, init=True)
                        T.set_flag("M", "FIX", L0C_FLAG)
                        T.wait_flag("M", "FIX", L0C_FLAG)
                        T.set_flag("M", "MTE2", Q_L1_FREE_FLAG)
                        T.copy(c_l0, QK_SLOT[cid, slot, :, g, :], enable_relu=True)
                        T.set_flag("FIX", "M", L0C_FLAG)
                    T.set_flag("M", "MTE2", K_L1_FREE_FLAG)
                    T.set_cross_flag("FIX", READY_FLAG + slot)

            with T.Scope("V"):
                mm_res_ub = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), calc_dtype)
                weight_ub = T.alloc_ub(VECTOR_BASEG, calc_dtype)
                reduce_tmp_ub = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), calc_dtype)
                reduce_g_ub = T.alloc_ub(VECTOR_BASEN, calc_dtype)
                history_ub = T.alloc_ub(2 * TOP_K, calc_dtype)
                sorted_block_ub = T.alloc_ub((1, 2 * BLOCK_N), calc_dtype)
                padded_trunk_ub = T.alloc_ub(2 * TOP_K, calc_dtype)
                merged_ub = T.alloc_ub(4 * TOP_K, calc_dtype)
                index_lane_ub = T.alloc_ub(2 * BLOCK_N, calc_dtype)
                selected_ub = T.alloc_ub(2 * TOP_K, calc_dtype)
                topk_index_ub = T.alloc_ub(TOP_K, calc_dtype)
                output_ub = T.alloc_ub(TOP_K, "int32")

                # 两个槽在首个 Cube 写入前均可复用。
                T.set_cross_flag("MTE2", FREE_FLAG)
                T.set_cross_flag("MTE2", FREE_FLAG + 1)
                T.set_flag("V", "MTE2", G_REDUCE_FLAG)
                T.tile.fill(index_lane_ub, 0)
                T.pipe_barrier("V")
                for index_offset in range(BLOCK_N):
                    index_lane_ub[index_offset * 2 + 1] = T.cast(1, calc_dtype)

                for n in T.serial(S2 // BLOCK_N):
                    slot = n % 2
                    T.wait_cross_flag(READY_FLAG + slot)
                    for s1_offset in T.serial(BLOCK_M):
                        s1_id = m_id * BLOCK_M + s1_offset
                        if n == 0:
                            T.tile.fill(history_ub, -T.infinity(calc_dtype))
                        else:
                            T.set_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                            T.wait_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                            T.copy(TOPK_HISTORY_WORKSPACE[cid, s1_offset, :], history_ub)
                            T.set_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                            T.wait_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                        T.tile.fill(reduce_tmp_ub, 0)
                        for g_id in T.serial(G // VECTOR_BASEG):
                            T.wait_flag("V", "MTE2", G_REDUCE_FLAG)
                            T.copy(
                                QK_SLOT[
                                    cid,
                                    slot,
                                    s1_offset,
                                    g_id * VECTOR_BASEG : (g_id + 1) * VECTOR_BASEG,
                                    0:VECTOR_BASEN,
                                ],
                                mm_res_ub,
                            )
                            T.copy(
                                WEIGHTS[
                                    b_id,
                                    s1_id,
                                    n2_id,
                                    g_id * VECTOR_BASEG : (g_id + 1) * VECTOR_BASEG,
                                ],
                                weight_ub,
                            )
                            T.set_flag("MTE2", "V", G_REDUCE_FLAG)
                            T.wait_flag("MTE2", "V", G_REDUCE_FLAG)
                            for i in T.serial(VECTOR_BASEG):
                                T.tile.mul(mm_res_ub[i, :], mm_res_ub[i, :], weight_ub[i])
                            T.tile.add(reduce_tmp_ub, mm_res_ub, reduce_tmp_ub)
                            T.set_flag("V", "MTE2", G_REDUCE_FLAG)
                        T.reduce_sum(reduce_tmp_ub, reduce_g_ub, 0)
                        T.tile.sort(sorted_block_ub, reduce_g_ub, BLOCK_N)
                        T.tile.axpy(sorted_block_ub, index_lane_ub, T.cast(n * BLOCK_N, calc_dtype))
                        T.pipe_barrier("V")
                        T.tile.fill(padded_trunk_ub, -T.infinity(calc_dtype))
                        T.pipe_barrier("V")
                        T.copy(sorted_block_ub[0, :], padded_trunk_ub[0 : 2 * BLOCK_N])
                        T.pipe_barrier("V")
                        T.tile.merge_sort(merged_ub, history_ub, padded_trunk_ub)
                        T.pipe_barrier("V")
                        T.copy(merged_ub[0 : 2 * TOP_K], history_ub)
                        T.set_flag("V", "MTE3", HISTORY_STORE_FLAG)
                        T.wait_flag("V", "MTE3", HISTORY_STORE_FLAG)
                        T.copy(history_ub, TOPK_HISTORY_WORKSPACE[cid, s1_offset, :])
                        T.set_flag("MTE3", "V", HISTORY_STORE_FLAG)
                        T.wait_flag("MTE3", "V", HISTORY_STORE_FLAG)
                    T.set_flag("V", "MTE2", SLOT_RELEASE_FLAG + slot)
                    T.wait_flag("V", "MTE2", SLOT_RELEASE_FLAG + slot)
                    T.set_cross_flag("MTE2", FREE_FLAG + slot)

                for s1_offset in T.serial(BLOCK_M):
                    s1_id = m_id * BLOCK_M + s1_offset
                    T.set_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                    T.wait_flag("V", "MTE2", HISTORY_LOAD_FLAG)
                    T.copy(TOPK_HISTORY_WORKSPACE[cid, s1_offset, :], history_ub)
                    T.set_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                    T.wait_flag("MTE2", "V", HISTORY_LOAD_FLAG)
                    T.copy(history_ub, selected_ub)
                    T.tile.gather_mask(topk_index_ub, selected_ub, "P1010")
                    T.tile.cast(output_ub, topk_index_ub, "CAST_ROUND", TOP_K)
                    T.set_flag("V", "MTE3", OUTPUT_READY_FLAG)
                    T.wait_flag("V", "MTE3", OUTPUT_READY_FLAG)
                    T.copy(output_ub, OUT[b_id, n2_id, s1_id, 0:TOP_K])
                    T.set_flag("MTE3", "V", OUTPUT_READY_FLAG)
                    T.wait_flag("MTE3", "V", OUTPUT_READY_FLAG)

    return main


def index_golden(q, k, weights, top_k):
    score_1 = torch.einsum("bsmgd, btmd->bmsgt", q, k).relu()
    score = score_1.permute(0, 2, 1, 3, 4)
    reduce_res = torch.sum(score * weights[..., None], dim=3)
    golden_out = torch.topk(reduce_res, top_k, dim=3, largest=True, sorted=True)
    return golden_out.indices.to(torch.int32).permute(0, 2, 1, 3)


def count_index_multiset_mismatches(expected, actual):
    if expected.shape != actual.shape:
        raise ValueError("输出索引张量形状不一致")
    total_mismatches = 0
    expected_rows = expected.reshape(-1, expected.shape[-1])
    actual_rows = actual.reshape(-1, actual.shape[-1])
    for expected_row, actual_row in zip(expected_rows, actual_rows):
        expected_counter = Counter(expected_row.tolist())
        actual_counter = Counter(actual_row.tolist())
        difference = (expected_counter - actual_counter) + (actual_counter - expected_counter)
        total_mismatches += sum(difference.values())
    return total_mismatches


def test_indexer(s1, s2, top_k, block_n=256, vector_basen=256):
    n2, groups, dimension = 1, 32, 64
    block_m, block_k = 64, 64
    vector_baseg = 16
    if s1 % block_m or s2 % block_n or groups % vector_baseg:
        raise ValueError("当前实现仅支持 S1、S2 和 G 分别整除 BLOCK_M、BLOCK_N、VECTOR_BASEG")
    if top_k > s2:
        raise ValueError("TOP_K 不能大于 S2")

    torch.manual_seed(2)
    func = indexer(
        n2,
        groups,
        dimension,
        top_k,
        vector_basen,
        vector_baseg,
        block_m,
        block_n,
        block_k,
        MAX_S2=s2,
    )
    query = torch.randn(1, s1, n2, groups, dimension).half()
    key = torch.randn(1, s2, n2, dimension).half()
    weights = torch.randn(1, s1, n2, groups).float()
    golden_out = index_golden(query, key, weights, top_k)

    torch.npu.synchronize()
    actual_out = func(query.view(1, s1, n2, -1).npu(), key.npu(), weights.npu()).cpu()
    torch.npu.synchronize()
    mismatches = count_index_multiset_mismatches(golden_out, actual_out)
    total_indices = s1 * n2 * top_k
    matched_ratio = 1 - mismatches / total_indices
    print(f"索引多重集合匹配率: {matched_ratio:.6f}，不匹配索引数: {mismatches}")
    if matched_ratio > 0.99:
        print("[PRECISION_PASS] 在线 TopK 索引多重集合匹配率超过 0.99")
        return
    print("[PRECISION_FAIL] 在线 TopK 索引多重集合匹配率未超过 0.99")
    raise AssertionError("在线 TopK 索引多重集合校验失败")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证 Lightning Indexer 在线 TopK")
    parser.add_argument("--quick", action="store_true", help="运行较小的整除形状用例")
    arguments = parser.parse_args()
    tilelang.disable_cache()
    if arguments.quick:
        test_indexer(s1=64, s2=1024, top_k=256, block_n=64, vector_basen=64)
    else:
        test_indexer(s1=512, s2=4096, top_k=1024)
