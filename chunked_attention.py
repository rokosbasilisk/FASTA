import torch
import triton
import triton.language as tl

@triton.jit
def chunk_attention_kernel(
    Q, K, V, Output,
    stride_qb, stride_qm, stride_qh,
    stride_kb, stride_km, stride_kh,
    stride_vb, stride_vm, stride_vh,
    stride_ob, stride_om, stride_oh,
    seq_len, chunk_size, head_dim,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Program IDs for parallel processing
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    
    chunk_start = pid_chunk * BLOCK_SIZE
    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, HEAD_DIM)
    mask = offs_m < (seq_len - chunk_start)
    
    # Load Q, K, V chunks
    q_base = Q + pid_batch * stride_qb + pid_head * stride_qh
    k_base = K + pid_batch * stride_kb + pid_head * stride_kh
    v_base = V + pid_batch * stride_vb + pid_head * stride_vh
    
    q_ptrs = q_base + chunk_start * stride_qm + offs_m[:, None] * stride_qm + offs_n[None, :] * 1
    k_ptrs = k_base + chunk_start * stride_km + offs_m[:, None] * stride_km + offs_n[None, :] * 1
    v_ptrs = v_base + chunk_start * stride_vm + offs_m[:, None] * stride_vm + offs_n[None, :] * 1
    
    q = tl.load(q_ptrs, mask=mask[:, None], other=0.0)
    k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask[:, None], other=0.0)
    
    # Intra-chunk attention
    scores = tl.dot(q, tl.trans(k)) * (1.0 / tl.sqrt(float(HEAD_DIM)))
    scores = tl.where(mask[:, None] & mask[None, :], scores, float("-inf"))
    
    # Apply Triton's softmax (last dimension only)
    max_scores = tl.max(scores, axis=1)
    scores = scores - max_scores[:, None]  # Subtract max for numerical stability
    exp_scores = tl.exp(scores)
    sum_exp_scores = tl.sum(exp_scores, axis=1)
    probs = exp_scores / sum_exp_scores[:, None]  # Normalize
    
    output = tl.dot(probs, v)
    
    # Store intra-chunk results
    o_base = Output + pid_batch * stride_ob + pid_head * stride_oh
    out_ptrs = o_base + chunk_start * stride_om + offs_m[:, None] * stride_om + offs_n[None, :] * 1
    tl.store(out_ptrs, output, mask=mask[:, None])

class ChunkedAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, chunk_size, use_mixed_precision=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert chunk_size >= 16, "chunk_size must be at least 16 for Triton"
        
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.use_mixed_precision = use_mixed_precision

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        dtype = torch.float16 if self.use_mixed_precision else torch.float32
        hidden_states = hidden_states.to(dtype)
        
        # Project and reshape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        output = torch.zeros_like(q, dtype=dtype)
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        # Triton kernel launch
        grid = (batch_size, self.num_heads, num_chunks)
        chunk_attention_kernel[grid](
            q, k, v, output,
            q.stride(0), q.stride(2), q.stride(1),
            k.stride(0), k.stride(2), k.stride(1),
            v.stride(0), v.stride(2), v.stride(1),
            output.stride(0), output.stride(2), output.stride(1),
            seq_len, self.chunk_size, self.head_dim,
            BLOCK_SIZE=min(128, self.chunk_size),
            HEAD_DIM=self.head_dim
        )
        
        # Final projection
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(output)
