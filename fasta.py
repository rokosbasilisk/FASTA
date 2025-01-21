import math
import torch
import triton
import triton.language as tl

def generate_minhash_indices(k, D, seed=0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randint(0, D, (k,), generator=rng, dtype=torch.int32)

def nearest_power_of_2(x):
    return max(16, 2 ** round(math.log2(x)))

@triton.jit
def fasta_minhash_tiled_kernel_no_smem(
    Q_ptr, K_ptr, Attn_ptr,
    minhash_indices_ptr,
    K_MINHASH: tl.constexpr,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr,

    # strides
    stride_qB, stride_qH, stride_qN, stride_qD,
    stride_kB, stride_kH, stride_kN, stride_kD,
    stride_aB, stride_aH, stride_aN, stride_aD
):
    pid = tl.program_id(0)
    n_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    b_idx = pid // (H * n_blocks * n_blocks)
    head_idx = (pid // (n_blocks * n_blocks)) % H
    block_idx = pid % (n_blocks * n_blocks)
    row_block_idx = block_idx // n_blocks
    col_block_idx = block_idx % n_blocks

    row_start = row_block_idx * BLOCK_SIZE
    col_start = col_block_idx * BLOCK_SIZE

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    offs_qN = row_start + tl.arange(0, BLOCK_SIZE)
    offs_kN = col_start + tl.arange(0, BLOCK_SIZE)

    if row_block_idx == col_block_idx:
        # Diagonal blocks => tile over [D]
        n_chunks = (D + BLOCK_K - 1) // BLOCK_K
        for cid in range(n_chunks):
            c_start = cid * BLOCK_K
            c_end   = tl.minimum(c_start + BLOCK_K, D)

            offs_dQ = c_start + tl.arange(0, BLOCK_K)
            offs_dK = c_start + tl.arange(0, BLOCK_K)
            valid_mask_dQ = offs_dQ < c_end
            valid_mask_dK = offs_dK < c_end

            q_ptrs = (Q_ptr
                + b_idx * stride_qB
                + head_idx * stride_qH
                + offs_qN[:, None] * stride_qN
                + offs_dQ[None, :] * stride_qD)
            k_ptrs = (K_ptr
                + b_idx * stride_kB
                + head_idx * stride_kH
                + offs_kN[:, None] * stride_kN
                + offs_dK[None, :] * stride_kD)

            q_mask2d = (offs_qN[:, None] < N) & valid_mask_dQ[None, :]
            k_mask2d = (offs_kN[:, None] < N) & valid_mask_dK[None, :]

            q_tile = tl.load(q_ptrs, mask=q_mask2d, other=0.0)
            k_tile = tl.load(k_ptrs, mask=k_mask2d, other=0.0)
            acc += tl.dot(q_tile, tl.trans(k_tile))

        acc *= 1.0 / tl.sqrt(float(D))
    else:
        # Off-diagonal => tile over k_minhash
        n_chunks = (K_MINHASH + BLOCK_K - 1) // BLOCK_K
        for cid in range(n_chunks):
            c_start = cid * BLOCK_K
            c_end   = tl.minimum(c_start + BLOCK_K, K_MINHASH)

            dims_idx = c_start + tl.arange(0, BLOCK_K)
            valid_mask_dims = dims_idx < c_end
            dims_chunk_ptr = minhash_indices_ptr + dims_idx
            dims_chunk = tl.load(dims_chunk_ptr, mask=valid_mask_dims, other=0)
            dims_2d = tl.reshape(dims_chunk, (1, BLOCK_K))

            row_offs_2d = tl.reshape(offs_qN, (BLOCK_SIZE, 1))
            col_offs_2d = tl.reshape(offs_kN, (BLOCK_SIZE, 1))

            q_ptrs = (Q_ptr
                + b_idx * stride_qB
                + head_idx * stride_qH
                + row_offs_2d * stride_qN
                + dims_2d * stride_qD)
            k_ptrs = (K_ptr
                + b_idx * stride_kB
                + head_idx * stride_kH
                + col_offs_2d * stride_kN
                + dims_2d * stride_kD)

            row_mask_q = (row_offs_2d < N)
            col_mask_k = (col_offs_2d < N)
            dim_mask   = (dims_2d < D) & valid_mask_dims[None, :]
            q_mask_2d  = row_mask_q & dim_mask
            k_mask_2d  = col_mask_k & dim_mask

            q_tile = tl.load(q_ptrs, mask=q_mask_2d, other=0.0)
            k_tile = tl.load(k_ptrs, mask=k_mask_2d, other=0.0)
            acc += tl.dot(q_tile, tl.trans(k_tile))

        # Scale by sqrt(D)/K_MINHASH to match diagonal magnitude
        acc *= tl.sqrt(float(D)) / float(K_MINHASH)

    # Store
    offs_i = row_start + tl.arange(0, BLOCK_SIZE)
    offs_j = col_start + tl.arange(0, BLOCK_SIZE)
    out_ptrs = (Attn_ptr
        + b_idx * stride_aB
        + head_idx * stride_aH
        + offs_i[:, None] * stride_aN
        + offs_j[None, :] * stride_aD)
    mask_out = (offs_i[:, None] < N) & (offs_j[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)

def fasta_minhash_tiled(Q, K, k_minhash=32, seed=42,block_size = 64, block_k = 16):
    B, H, N, D = Q.shape

    dims = generate_minhash_indices(k_minhash, D, seed).to(Q.device)
    attn = torch.zeros((B, H, N, N), device=Q.device, dtype=Q.dtype)
    n_blocks = (N + block_size - 1) // block_size
    grid = (B * H * n_blocks * n_blocks,)

    Qc = Q.contiguous()
    Kc = K.contiguous()

    
    fasta_minhash_tiled_kernel_no_smem[grid](
        Qc, Kc, attn,
        dims,
        k_minhash,
        B, H, N, D,
        block_size,
        block_k,
        Qc.stride(0), Qc.stride(1), Qc.stride(2), Qc.stride(3),
        Kc.stride(0), Kc.stride(1), Kc.stride(2), Kc.stride(3),
        attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
    )
    return attn
