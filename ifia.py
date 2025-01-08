import torch
import triton
import triton.language as tl

@triton.jit
def compute_chunk_mean(
    X, Mean,
    stride_b, stride_h, stride_m,
    seq_len, chunk_size, head_dim,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Program IDs for parallel processing
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    
    # Calculate chunk start
    chunk_start = pid_chunk * BLOCK_SIZE
    
    # Create offsets and mask
    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, HEAD_DIM)
    mask = offs_m < (seq_len - chunk_start)
    
    # Load chunk
    base_ptr = X + pid_batch * stride_b + pid_head * stride_h
    x_ptrs = base_ptr + chunk_start * stride_m + offs_m[:, None] * stride_m + offs_n[None, :] * 1
    x = tl.load(x_ptrs, mask=mask[:, None], other=0.0)
    
    # Compute mean
    chunk_sum = tl.sum(x, 0)
    valid_elements = tl.sum(mask.to(tl.float32))
    chunk_mean = chunk_sum / valid_elements
    
    # Store mean
    mean_ptr = Mean + pid_batch * head_dim * pid_chunk + pid_head * head_dim + offs_n
    tl.store(mean_ptr, chunk_mean)

@triton.jit
def intra_chunk_attention(
    Q, K, V, Output,
    stride_qb, stride_qm, stride_qh,
    stride_kb, stride_km, stride_kh,
    stride_vb, stride_vm, stride_vh,
    stride_ob, stride_om, stride_oh,
    seq_len, chunk_size, head_dim,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
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
    
    # Compute attention
    scores = tl.dot(q, tl.trans(k))
    scores = scores * (1.0 / tl.sqrt(float(HEAD_DIM)))
    scores = tl.where(mask[:, None] & mask[None, :], scores, float("-inf"))
    
    # Softmax
    scores = tl.exp(scores - tl.max(scores, 1)[:, None])
    scores = scores / tl.sum(scores, 1)[:, None]
    
    # Final computation
    output = tl.dot(scores, v)
    
    # Store results
    o_base = Output + pid_batch * stride_ob + pid_head * stride_oh
    out_ptrs = o_base + chunk_start * stride_om + offs_m[:, None] * stride_om + offs_n[None, :] * 1
    tl.store(out_ptrs, output, mask=mask[:, None])

class ChunkedAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, chunk_size):
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

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project and reshape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        output = torch.zeros_like(q)
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        # Step 1: Process intra-chunk attention
        grid = (batch_size, self.num_heads, num_chunks)
        intra_chunk_attention[grid](
            q, k, v, output,
            q.stride(0), q.stride(2), q.stride(1),
            k.stride(0), k.stride(2), k.stride(1),
            v.stride(0), v.stride(2), v.stride(1),
            output.stride(0), output.stride(2), output.stride(1),
            seq_len, self.chunk_size, self.head_dim,
            BLOCK_SIZE=min(128, self.chunk_size),
            HEAD_DIM=self.head_dim
        )
        
        # Step 2: Compute chunk means
        q_means = torch.zeros(batch_size, self.num_heads, num_chunks, self.head_dim, device=q.device)
        k_means = torch.zeros_like(q_means)
        
        grid = (batch_size, self.num_heads, num_chunks)
        compute_chunk_mean[grid](
            q, q_means,
            q.stride(0), q.stride(1), q.stride(2),
            seq_len, self.chunk_size, self.head_dim,
            BLOCK_SIZE=min(128, self.chunk_size),
            HEAD_DIM=self.head_dim
        )
        compute_chunk_mean[grid](
            k, k_means,
            k.stride(0), k.stride(1), k.stride(2),
            seq_len, self.chunk_size, self.head_dim,
            BLOCK_SIZE=min(128, self.chunk_size),
            HEAD_DIM=self.head_dim
        )
        
        # Step 3: Process proper inter-chunk attention
        # Reshape tensors into chunks
        q_chunks = q.view(batch_size, self.num_heads, num_chunks, self.chunk_size, self.head_dim)
        k_chunks = k.view(batch_size, self.num_heads, num_chunks, self.chunk_size, self.head_dim)
        v_chunks = v.view(batch_size, self.num_heads, num_chunks, self.chunk_size, self.head_dim)
        
        # Compute chunk summaries (means)
        k_means = torch.mean(k_chunks, dim=3)  # [batch, heads, num_chunks, head_dim]
        v_means = torch.mean(v_chunks, dim=3)  # [batch, heads, num_chunks, head_dim]
        
        # For each query token, compute attention with chunk means
        # Shape: [batch, heads, num_chunks, chunk_size, num_chunks]
        inter_chunk_scores = torch.matmul(
            q_chunks,  # [batch, heads, num_chunks, chunk_size, head_dim]
            k_means.transpose(-2, -1).unsqueeze(2)  # [batch, heads, 1, head_dim, num_chunks]
        ) / (self.head_dim ** 0.5)
        
        # Apply causal masking for inter-chunk attention
        chunk_positions = torch.arange(num_chunks, device=q.device)
        causal_mask = (chunk_positions[:, None] >= chunk_positions[None, :])  # [num_chunks, num_chunks]
        causal_mask = causal_mask.view(1, 1, num_chunks, 1, num_chunks)
        causal_mask = causal_mask.expand(batch_size, self.num_heads, num_chunks, self.chunk_size, num_chunks)
        inter_chunk_scores = inter_chunk_scores.masked_fill(~causal_mask, float('-inf'))
        
        # Softmax across chunks
        inter_chunk_attn = torch.softmax(inter_chunk_scores, dim=-1)
        
        # Apply attention to value means
        # [batch, heads, num_chunks, chunk_size, num_chunks] x [batch, heads, num_chunks, head_dim]
        chunk_output = torch.einsum('bhqtc,bhcd->bhqtd', inter_chunk_attn, v_means)
        
        # Reshape back to original dimensions and add to output
        chunk_output = chunk_output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        output = output + chunk_output
        
        # Final projection
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(output)

def test_chunked_attention():
    # Test parameters
    embed_dim = 64
    num_heads = 4
    chunk_size = 32
    seq_len = 128
    batch_size = 2

    # Create random input
    hidden_states = torch.randn(batch_size, seq_len, embed_dim, device="cuda")
    
    # Initialize attention layer
    attention_layer = ChunkedAttention(embed_dim, num_heads, chunk_size).cuda()
    
    # Forward pass
    output = attention_layer(hidden_states)
    
    # Verify output shape
    print("Output shape:", output.shape)
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output.shape == expected_shape, f"Output shape mismatch. Expected {expected_shape}, got {output.shape}"
    print("Test passed successfully!")

if __name__ == "__main__":
    test_chunked_attention()
