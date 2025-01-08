# FASTA: Full Average Scaled Tiling Attention
implement a sparse attention using triton using the following methods
in the standard self attention, the attention weight is computed like this: attn_weight = query @ key.transpose(-2, -1) * scale_factor
assume a function:
def att_weight(Q,K_T):
    return Q@K_T
FASTA is a sparse approximation for the above function which works as follows:
the Q and K are divided into equal sized chunks
assume  QxK^T to be [Q0,Q1,....Qn-1]*[K0,K1,....Kn-1] where each of them are equal sized chunks from the initial embeddings.
in the full product if Q0*K0 then you do the regular multiplication, but if Q0*K1 or whenever the indices are not same, do avg(Q0)*avg(K1) and then broadcast this value in the shape of that grid.
create a triton kernel which implements the above operation if i==j then intra-index, if i!=j then inter-index
generate code and test case for the kernels first before proceeding to the full implementation
