import torch
import triton
import triton.language as tl
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # For progress bar

@triton.jit
def fasta_kernel(
    Q_ptr, K_ptr, attn_ptr,
    B, H, N, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    stride_qb, stride_qh, stride_q0, stride_q1,
    stride_kb, stride_kh, stride_k0, stride_k1,
    stride_attnb, stride_attnh, stride_attn0, stride_attn1,
):
    # Get the program ID and compute batch, head, row, and column block indices
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)
    b_idx = pid // (H * n_blocks * n_blocks)
    head_idx = (pid // (n_blocks * n_blocks)) % H
    block_idx = pid % (n_blocks * n_blocks)
    row_block_idx = block_idx // n_blocks
    col_block_idx = block_idx % n_blocks

    # Calculate the starting positions of the current block
    row_start = row_block_idx * BLOCK_SIZE
    col_start = col_block_idx * BLOCK_SIZE

    # Create block offsets for Q and K
    offs_q = row_start + tl.arange(0, BLOCK_SIZE)
    offs_k = col_start + tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, D)

    # Initialize the accumulator for attention weights
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Compute memory addresses for Q and K blocks
    q_ptrs = Q_ptr + b_idx * stride_qb + head_idx * stride_qh + offs_q[:, None] * stride_q0 + offs_d[None, :] * stride_q1
    k_ptrs = K_ptr + b_idx * stride_kb + head_idx * stride_kh + offs_k[:, None] * stride_k0 + offs_d[None, :] * stride_k1

    # Create masks to handle boundary conditions
    q_mask = offs_q[:, None] < N
    k_mask = offs_k[:, None] < N

    # Load Q and K blocks with masking
    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)  # Shape: (BLOCK_SIZE, D)
    k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)  # Shape: (BLOCK_SIZE, D)

    # Intra-block attention: exact computation using block-wise matmul
    if row_block_idx == col_block_idx:
        acc += tl.dot(q_block, tl.trans(k_block))

    # Inter-block attention:
    if row_block_idx != col_block_idx:
        q_vector = tl.sum(q_block, axis=1)  # Shape: (BLOCK_SIZE,)
        k_vector = tl.sum(k_block, axis=1)  # Shape: (BLOCK_SIZE,)
        outer = q_vector[:, None] * k_vector[None, :]
        acc += outer / (D ** 2)

    # Calculate offsets for storing the attention weights
    offs_attn_i = row_start + tl.arange(0, BLOCK_SIZE)
    offs_attn_j = col_start + tl.arange(0, BLOCK_SIZE)

    # Compute memory addresses for storing the attention weights
    attn_ptrs = attn_ptr + b_idx * stride_attnb + head_idx * stride_attnh + offs_attn_i[:, None] * stride_attn0 + offs_attn_j[None, :] * stride_attn1

    # Create a mask to handle boundary conditions during storage
    mask = (offs_attn_i[:, None] < N) & (offs_attn_j[None, :] < N)

    # Store the computed attention weights with masking
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
    # Ensure tensors are contiguous
    Q = Q.contiguous()
    K = K.contiguous()

    # Create output tensor
    attn = torch.empty((B, H, N, N), device=Q.device, dtype=Q.dtype)

    # Calculate grid size
    n_blocks = triton.cdiv(N, block_size)
    grid = (B * H * n_blocks * n_blocks,)

    # Launch kernel
    fasta_kernel[grid](
        Q, K, attn,
        B, H, N, D, block_size,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
    )

    return attn

