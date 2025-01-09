import torch
import triton
import triton.language as tl

@triton.jit
def fasta_kernel(
    Q_ptr, K_ptr, attn_ptr,
    B, H, N, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    stride_qb, stride_qh, stride_q0, stride_q1,
    stride_kb, stride_kh, stride_k0, stride_k1,
    stride_attnb, stride_attnh, stride_attn0, stride_attn1,
    locality_threshold: tl.constexpr
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

    q_mask = offs_q[:, None] < N
    k_mask = offs_k[:, None] < N

    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)
    k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

    # Cache for the most recent diagonal block computation
    diagonal_cache = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    diagonal_mean = 0.0  # Initialize as scalar
    diagonal_var = 0.0   # Initialize as scalar

    if tl.abs(row_block_idx - col_block_idx) < locality_threshold:
        block_attn = tl.dot(q_block, tl.trans(k_block))
        acc += block_attn

        # Store diagonal statistics in cache
        diagonal_mean = tl.sum(block_attn) / (BLOCK_SIZE * BLOCK_SIZE)
        diagonal_var = tl.sum((block_attn - diagonal_mean) * (block_attn - diagonal_mean)) / (BLOCK_SIZE * BLOCK_SIZE)
        diagonal_cache = block_attn
    else:
        q_mask_sum = tl.sum(q_mask)
        q_mask_sum = tl.where(q_mask_sum == 0, 1.0, q_mask_sum)
        q_summary = tl.sum(q_block, axis=0) / q_mask_sum

        k_mask_sum = tl.sum(k_mask)
        k_mask_sum = tl.where(k_mask_sum == 0, 1.0, k_mask_sum)
        k_summary = tl.sum(k_block, axis=0) / k_mask_sum

        dot_product = tl.sum(q_summary * k_summary)
        approx_attn = tl.full((BLOCK_SIZE, BLOCK_SIZE), dot_product, dtype=tl.float32)

        # Normalize and scale based on cached diagonal block statistics
        approx_attn = (approx_attn - diagonal_mean) / tl.sqrt(diagonal_var + 1e-6)

        valid_mask = q_mask & k_mask.T
        approx_attn = tl.where(valid_mask, approx_attn, 0.0)
        acc += approx_attn

    offs_attn_i = row_start + tl.arange(0, BLOCK_SIZE)
    offs_attn_j = col_start + tl.arange(0, BLOCK_SIZE)
    attn_ptrs = attn_ptr + b_idx * stride_attnb + head_idx * stride_attnh + offs_attn_i[:, None] * stride_attn0 + offs_attn_j[None, :] * stride_attn1
    mask = (offs_attn_i[:, None] < N) & (offs_attn_j[None, :] < N)
    tl.store(attn_ptrs, acc, mask=mask)


def fasta_attn(Q, K, block_size, locality_threshold=2):
    """
    Computes FASTA attention using Triton with optimized intra-block matmul and Gaussian-like spread.

    Args:
        Q (torch.Tensor): Query tensor of shape (B, H, N, D)
        K (torch.Tensor): Key tensor of shape (B, H, N, D)
        block_size (int): Size of attention blocks
        locality_threshold (int): Threshold for locality comparison

    Returns:
        torch.Tensor: Attention weights of shape (B, H, N, N)
    """
    B, H, N, D = Q.shape

    Q = Q.contiguous()
    K = K.contiguous()

    attn = torch.empty((B, H, N, N), device=Q.device, dtype=Q.dtype)

    n_blocks = triton.cdiv(N, block_size)
    grid = (B * H * n_blocks * n_blocks,)

    fasta_kernel[grid](
        Q, K, attn,
        B, H, N, D, block_size,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
        locality_threshold
    )

    return attn
