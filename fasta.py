import torch
import triton
import triton.language as tl
import math

def nearest_power_of_2(x):
    """Find the nearest power of 2 greater than or equal to x."""
    return max(16, 2 ** round(math.log2(x)))

@triton.jit
def fasta_kernel(
    Q_ptr, K_ptr, attn_ptr,
    B, H, N, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    stride_qb, stride_qh, stride_q0, stride_q1,
    stride_kb, stride_kh, stride_k0, stride_k1,
    stride_attnb, stride_attnh, stride_attn0, stride_attn1
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

    if row_block_idx == col_block_idx:
        q_ptrs = Q_ptr + b_idx * stride_qb + head_idx * stride_qh + offs_q[:, None] * stride_q0 + offs_d[None, :] * stride_q1
        k_ptrs = K_ptr + b_idx * stride_kb + head_idx * stride_kh + offs_k[:, None] * stride_k0 + offs_d[None, :] * stride_k1

        q_mask = offs_q[:, None] < N
        k_mask = offs_k[:, None] < N

        q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Ensure valid dimensions for tl.dot
        assert q_block.shape[-1] >= 16 and k_block.shape[-1] >= 16, \
            f"Block dimensions are too small: {q_block.shape}, {k_block.shape}"

        block_attn = tl.dot(q_block, tl.trans(k_block))
        acc += block_attn

    offs_attn_i = row_start + tl.arange(0, BLOCK_SIZE)
    offs_attn_j = col_start + tl.arange(0, BLOCK_SIZE)
    attn_ptrs = attn_ptr + b_idx * stride_attnb + head_idx * stride_attnh + offs_attn_i[:, None] * stride_attn0 + offs_attn_j[None, :] * stride_attn1
    mask = (offs_attn_i[:, None] < N) & (offs_attn_j[None, :] < N) & (row_block_idx == col_block_idx)
    tl.store(attn_ptrs, acc, mask=mask)

def fasta_attn(Q, K):
    """
    Computes FASTA attention using Triton with optimized intra-block matmul.

    Args:
        Q (torch.Tensor): Query tensor of shape (B, H, N, D)
        K (torch.Tensor): Key tensor of shape (B, H, N, D)

    Returns:
        torch.Tensor: Attention weights of shape (B, H, N, N)
    """
    B, H, N, D = Q.shape

    # Adjust block_size to the nearest power of 2 to sqrt(N)
    block_size = min(nearest_power_of_2(math.sqrt(N)), N)

    Q = Q.contiguous()
    K = K.contiguous()

    attn = torch.zeros((B, H, N, N), device=Q.device, dtype=Q.dtype)

    n_blocks = triton.cdiv(N, block_size)
    grid = (B * H * n_blocks * n_blocks,)

    fasta_kernel[grid](
        Q, K, attn,
        B, H, N, D, block_size,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3)
    )
    return attn

