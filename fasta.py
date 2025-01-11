import torch
import triton
import triton.language as tl

# Precompute random projection matrix on the host
def precompute_random_projection(D, K_REDUCED):
    torch.manual_seed(42)  # For reproducibility
    return torch.randn(D, K_REDUCED, dtype=torch.float32, device="cuda")  # Precompute projection matrix

def precompute_projected_qk(Q, K, rand_proj):
    Q_proj = torch.matmul(Q, rand_proj)  # Precompute projected Q
    K_proj = torch.matmul(K, rand_proj)  # Precompute projected K
    return Q_proj, K_proj

@triton.jit
def optimized_fasta_kernel(
    Q_ptr, K_ptr, attn_ptr, rand_proj_ptr,
    B, H, N, D: tl.constexpr, K_REDUCED: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    stride_qb, stride_qh, stride_q0, stride_q1,
    stride_kb, stride_kh, stride_k0, stride_k1,
    stride_attnb, stride_attnh, stride_attn0, stride_attn1,
    stride_rand0, stride_rand1
):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)
    b_idx = pid // (H * n_blocks * n_blocks)
    head_idx = (pid // (n_blocks * n_blocks)) % H
    block_idx = pid % (n_blocks * n_blocks)
    row_block_idx = block_idx // n_blocks
    col_block_idx = block_idx % n_blocks

    row_start = row_block_idx * BLOCK_SIZE
    col_start = col_block_idx * BLOCK_SIZE

    offs_q = row_start + tl.arange(0, BLOCK_SIZE)
    offs_k = col_start + tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, D)

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    q_ptrs = Q_ptr + b_idx * stride_qb + head_idx * stride_qh + offs_q[:, None] * stride_q0 + offs_d[None, :] * stride_q1
    k_ptrs = K_ptr + b_idx * stride_kb + head_idx * stride_kh + offs_k[:, None] * stride_k0 + offs_d[None, :] * stride_k1

    rand_proj_ptrs = rand_proj_ptr + offs_d[:, None] * stride_rand0 + tl.arange(0, K_REDUCED)[None, :] * stride_rand1
    rand_proj = tl.load(rand_proj_ptrs, mask=offs_d[:, None] < D, other=0.0)

    q_mask = offs_q[:, None] < N
    k_mask = offs_k[:, None] < N

    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)
    k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

    if row_block_idx == col_block_idx:
        # Compute exact attention for diagonal blocks
        exact_dot_product = tl.dot(q_block, tl.trans(k_block))
        acc += exact_dot_product
    else:
        # Project Q and K into reduced space for off-diagonal blocks
        norm_factor = tl.sqrt(tl.full([], K_REDUCED, dtype=tl.float32))
        q_projected = tl.dot(q_block, rand_proj) / norm_factor
        k_projected = tl.dot(k_block, rand_proj) / norm_factor

        # Compute approximate attention
        reduced_dot_product = tl.dot(q_projected, tl.trans(k_projected))
        acc += reduced_dot_product

    offs_attn_i = row_start + tl.arange(0, BLOCK_SIZE)
    offs_attn_j = col_start + tl.arange(0, BLOCK_SIZE)
    attn_ptrs = attn_ptr + b_idx * stride_attnb + head_idx * stride_attnh + offs_attn_i[:, None] * stride_attn0 + offs_attn_j[None, :] * stride_attn1
    mask = (offs_attn_i[:, None] < N) & (offs_attn_j[None, :] < N)
    tl.store(attn_ptrs, acc, mask=mask)


def fasta_attn(Q, K, block_size):
    B, H, N, D = Q.shape
    K_REDUCED = 16  # Fixed reduced dimensionality

    # Precompute random projection matrix
    rand_proj = precompute_random_projection(D, K_REDUCED)

    Q = Q.contiguous()
    K = K.contiguous()

    attn = torch.empty((B, H, N, N), device=Q.device, dtype=Q.dtype)

    n_blocks = triton.cdiv(N, block_size)
    grid = (B * H * n_blocks * n_blocks,)

    optimized_fasta_kernel[grid](
        Q, K, attn, rand_proj,
        B, H, N, D, K_REDUCED, block_size,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
        rand_proj.stride(0), rand_proj.stride(1)
    )

    return attn


