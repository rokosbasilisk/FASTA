import torch
import triton
import triton.language as tl

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

    q_ptrs = Q_ptr + b_idx * stride_qb + head_idx * stride_qh + offs_q[:, None] * stride_q0 + offs_d[None, :] * stride_q1
    k_ptrs = K_ptr + b_idx * stride_kb + head_idx * stride_kh + offs_k[:, None] * stride_k0 + offs_d[None, :] * stride_k1

    q_mask = offs_q[:, None] < N
    k_mask = offs_k[:, None] < N

    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)
    k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

    if True: #row_block_idx == col_block_idx:
        block_attn = tl.dot(q_block, tl.trans(k_block))
        acc += block_attn
    else:
        distance = tl.abs(row_block_idx - col_block_idx)
        sampled_indices = D // (2 * distance)
        sampled_indices = max(1, sampled_indices)  # Ensure at least one index is sampled

        # Manually sample indices by constructing a sampled mask
        sampled_mask = tl.arange(0, D) % max(D // sampled_indices, 1) == 0
        q_sampled = tl.load(q_ptrs, mask=sampled_mask[None, :], other=0.0)
        k_sampled = tl.load(k_ptrs, mask=sampled_mask[None, :], other=0.0)

        dot_product = tl.sum(q_sampled * k_sampled, axis=1)
        sampled_mean = tl.sum(dot_product) / sampled_indices  # Compute mean explicitly
        approx_attn = tl.full((BLOCK_SIZE, BLOCK_SIZE), sampled_mean, dtype=tl.float32)

        valid_mask = q_mask & k_mask.T
        approx_attn = tl.where(valid_mask, approx_attn, 0.0)
        acc += approx_attn

    offs_attn_i = row_start + tl.arange(0, BLOCK_SIZE)
    offs_attn_j = col_start + tl.arange(0, BLOCK_SIZE)
    attn_ptrs = attn_ptr + b_idx * stride_attnb + head_idx * stride_attnh + offs_attn_i[:, None] * stride_attn0 + offs_attn_j[None, :] * stride_attn1
    mask = (offs_attn_i[:, None] < N) & (offs_attn_j[None, :] < N)
    tl.store(attn_ptrs, acc, mask=mask)



def fasta_attn(Q, K, block_size):
    """
    Computes FASTA attention using Triton with optimized intra-block matmul and Gaussian-like spread.

    Args:
        Q (torch.Tensor): Query tensor of shape (B, H, N, D)
        K (torch.Tensor): Key tensor of shape (B, H, N, D)
        block_size (int): Size of attention blocks

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
        attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3)
    )

    return attn
