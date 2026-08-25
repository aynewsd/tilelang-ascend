import torch
from collections import Counter

torch.manual_seed(2)
import tilelang
import tilelang.language as T

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], workspace_idx=[-3], pass_configs=pass_configs, target="ascendc")
def indexer(N2, G, D, TOP_K, VECTOR_BASEN, VECTOR_BASEG, BLOCK_M, BLOCK_N, BLOCK_K, MAX_S2=4096, input_dtype="float16", calc_dtype="float"):

    B = T.symbolic("B")
    S1 = T.symbolic("S1")
    S2 = T.symbolic("S2")

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N2, G * D), input_dtype),
        KEY: T.Tensor((B, S2, N2, D), input_dtype),
        QK_RES: T.Tensor((B, N2, S1, G, S2), calc_dtype),
        WEIGHTS: T.Tensor((B, S1, N2, G), calc_dtype),
        OUT: T.Tensor((B, N2, S1, TOP_K), "int"),
    ):

        S1_TILES = S1 // BLOCK_M

        with T.Kernel(B * N2 * S1_TILES, is_npu=True) as (cid, vid):
            b_id = cid // (N2 * S1_TILES)
            n2_id = (cid // S1_TILES) % N2
            m_id = cid % S1_TILES

            with T.Scope("C"):
                Q_L1 = T.alloc_L1((BLOCK_M, BLOCK_K), input_dtype)
                K_L1 = T.alloc_L1((BLOCK_N, BLOCK_K), input_dtype)
                C_L0 = T.alloc_L0C((BLOCK_M, BLOCK_N), calc_dtype)

                for n in T.serial(S2 // BLOCK_N):
                    T.copy(KEY[b_id, n * BLOCK_N : (n + 1) * BLOCK_N, n2_id, 0:D], K_L1)
                    for g in T.serial(G):
                        T.copy(Query[b_id, m_id * BLOCK_M : (m_id + 1) * BLOCK_M, n2_id, g * D : (g + 1) * D], Q_L1)
                        T.gemm_v0(Q_L1, K_L1, C_L0, transpose_B=True, init=True)
                        T.copy(
                            C_L0,
                            QK_RES[
                                b_id, n2_id, m_id * BLOCK_M : (m_id + 1) * BLOCK_M, g, n * BLOCK_N : (n + 1) * BLOCK_N
                            ],
                            enable_relu=True,
                        )
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                mm_res_ub = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), calc_dtype)
                weight_ub = T.alloc_ub(VECTOR_BASEG, calc_dtype)
                reduce_tmp_ub = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), calc_dtype)
                reduce_g_ub = T.alloc_ub(VECTOR_BASEN, calc_dtype)
                score_accum_ub = T.alloc_ub(MAX_S2, calc_dtype)
                topk_dst_ub = T.alloc_ub(2 * TOP_K, calc_dtype)
                topk_index_ub = T.alloc_ub(TOP_K, calc_dtype)
                output_ub = T.alloc_ub(TOP_K, "int")

                T.wait_cross_flag(0)
                for s1_offset in T.serial(BLOCK_M):
                    s1_id = m_id * BLOCK_M + s1_offset
                    T.tile.fill(score_accum_ub, 0)
                    for s2_id in T.serial(S2 // VECTOR_BASEN):
                        T.tile.fill(reduce_tmp_ub, 0)
                        T.tile.fill(reduce_g_ub, 0)

                        for g_id in T.serial(G // VECTOR_BASEG):
                            T.copy(
                                QK_RES[
                                    b_id,
                                    n2_id,
                                    s1_id,
                                    g_id * VECTOR_BASEG : (g_id + 1) * VECTOR_BASEG,
                                    s2_id * VECTOR_BASEN : (s2_id + 1) * VECTOR_BASEN,
                                ],
                                mm_res_ub,
                            )
                            T.copy(WEIGHTS[b_id, s1_id, n2_id, g_id * VECTOR_BASEG : (g_id + 1) * VECTOR_BASEG], weight_ub)
                            for i in range(VECTOR_BASEG):
                                T.tile.mul(mm_res_ub[i, :], mm_res_ub[i, :], weight_ub[i])
                            T.tile.add(reduce_tmp_ub, mm_res_ub, reduce_tmp_ub)
                        T.reduce_sum(reduce_tmp_ub, reduce_g_ub, 0)
                        for i in T.serial(VECTOR_BASEN):
                            score_accum_ub[s2_id * VECTOR_BASEN + i] = reduce_g_ub[i]
                    T.tile.topk(topk_dst_ub, score_accum_ub, TOP_K, S2)
                    T.tile.gather_mask(topk_index_ub, topk_dst_ub, "P1010")
                    T.tile.cast(output_ub, topk_index_ub, "CAST_ROUND", TOP_K)
                    T.copy(output_ub, OUT[b_id, n2_id, s1_id, 0:TOP_K])

    return main


N2 = 1
G = 32
D = 64
TOP_K = 1024


def index_golden(q, k, weights):
    score_1 = torch.einsum("bsmgd, btmd->bmsgt", q, k)
    score_1 = score_1.relu()
    score = score_1.permute(0, 2, 1, 3, 4)
    mul_res = score * weights
    reduce_res = torch.sum(mul_res, dim=3)
    golden_out = torch.topk(reduce_res, TOP_K, dim=3, largest=True, sorted=True)
    return score_1.float(), golden_out.indices.to(torch.int32).permute(0, 2, 1, 3)


def count_mismatches_last_dim(tensor1, tensor2):
    assert tensor1.shape[-1] == tensor2.shape[-1], "the last dimension of two tensors must be the same"
    last_dim = tensor1.shape[-1]
    tensor1_flat = tensor1.view(-1, last_dim)
    tensor2_flat = tensor2.view(-1, last_dim)

    total_mismatches = 0

    for i in range(tensor1_flat.shape[0]):
        row1 = tensor1_flat[i].tolist()
        row2 = tensor2_flat[i].tolist()

        counter1 = Counter(row1)
        counter2 = Counter(row2)

        diff = (counter1 - counter2) + (counter2 - counter1)
        total_mismatches += sum(diff.values())

    return total_mismatches


def test_indexer():
    B = 1
    S1 = 512
    S2 = 4096
    func = indexer(N2, G, D, TOP_K, 256, 16, 64, 64, 64)

    q = torch.randn(B, S1, N2, G, D).half()
    k = torch.randn(B, S2, N2, D).half()
    weights = torch.randn(B, S1, N2, G, 1).float()

    qk_res_workspace_, golden_out = index_golden(q, k, weights)

    q_npu = q.view(B, S1, N2, -1).npu()
    k_npu = k.npu()
    weights_npu = weights.npu()
    torch.npu.synchronize()
    npu_out = func(q_npu, k_npu, weights_npu).to(torch.int32)
    torch.npu.synchronize()

    total_mismatches = count_mismatches_last_dim(golden_out.cpu(), npu_out.cpu())

    if (1 - total_mismatches / (B * S1 * N2 * TOP_K)) > 0.99:
        print("Test passed!")
    else:
        print("Test failed! The precision is not correct!")


if __name__ == "__main__":
    test_indexer()
