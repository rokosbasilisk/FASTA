# FASTA: Full Average Scaled Tiling Attention
- implement a sparse attention using triton using the following methods
- in the standard self attention, the attention weight is computed like this: attn_weight = query @ key.transpose(-2, -1) * scale_factor which takes O(n^2d) time
- in FASTA the Q and K are divided into equal sized chunks [Q0,Q1,....Qn-1]*[K0,K1,....Kn-1] where each of them are equal sized chunks
- in the full product if Q0*K0 then you do the regular multiplication, but if Q0*K1 or whenever the indices are not same, do avg(Q0)*avg(K1) and then broadcast this value in the shape of that grid.
