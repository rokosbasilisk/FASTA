# FASTA: Full Average Scaled Tiling Attention

This project implements a sparse attention mechanism using Triton. The attention computation follows these methods:

### Standard Self-Attention
In standard self-attention, the attention weights are computed as follows:
\[
\text{attn\_weight} = \text{query} \cdot \text{key}^\top \cdot \text{scale\_factor}
\]
This process has a time complexity of \( O(n^2d) \), where \( n \) is the sequence length, and \( d \) is the hidden dimension.

### FASTA Mechanism
In the FASTA mechanism:
1. The **query (Q)** and **key (K)** matrices are divided into equal-sized chunks:
   \[
   [Q_0, Q_1, \dots, Q_{n-1}] \quad \text{and} \quad [K_0, K_1, \dots, K_{n-1}]
   \]
   Each chunk is of equal size.

2. For each chunk pair \( Q_i \cdot K_j \):
   - If \( i = j \) (i.e., the indices are the same), the **regular multiplication** is performed:
     \[
     Q_i \cdot K_i^\top
     \]
   - If \( i \neq j \) (i.e., the indices are different), the average of \( Q_i \) and \( K_j \) is computed:
     \[
     \text{avg}(Q_i) \cdot \text{avg}(K_j)
     \]
     This single scalar value is then **broadcast** to fill the shape of the corresponding grid.

This approach significantly reduces the computational complexity by replacing many full matrix multiplications with scalar computations and broadcasting.
