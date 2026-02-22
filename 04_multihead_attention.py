"""
Stage 4: Multi-Head Attention
------------------------------
Instead of one attention head, we run several in parallel.
Each head independently learns different relationship types:
  - One head might track subject-verb agreement
  - Another might track coreference ("it" → "animal")
  - Another might track word order patterns

Architecture:
  embed_dim=256, num_heads=4 → each head sees head_dim=64
  4 heads × 64 dims = 256 → concatenate → linear projection → 256
"""

import torch
import torch.nn as nn

torch.manual_seed(42)


# ---------------------------------------------------------------
# STEP 1: WHY MULTIPLE HEADS?
# ---------------------------------------------------------------
print("=" * 50)
print("STEP 1: WHY MULTIPLE HEADS?")
print("=" * 50)
print("""
Single head: one set of W_q, W_k, W_v → learns ONE relationship type.

Multi-head: N independent sets of W_q, W_k, W_v → N relationship types.

Analogy: reading a sentence and simultaneously asking:
  Head 1: "which words are grammatically related?"
  Head 2: "which words refer to the same entity?"
  Head 3: "which words are semantically similar?"
  Head 4: "which words are positionally close?"

Each head gets a 64-dim slice to work with (instead of 256),
so the total compute stays the same as one full-dim head.
""")


# ---------------------------------------------------------------
# STEP 2: NAIVE IMPLEMENTATION (loop over heads)
# ---------------------------------------------------------------
print("=" * 50)
print("STEP 2: NAIVE MULTI-HEAD (loop, easy to understand)")
print("=" * 50)

class MultiHeadAttentionNaive(nn.Module):
    """
    Simple version: run each head separately in a loop.
    Correct but slow — shown for clarity.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # One set of Q, K, V weights per head
        self.heads = nn.ModuleList([
            nn.ModuleDict({
                'W_q': nn.Linear(embed_dim, self.head_dim, bias=False),
                'W_k': nn.Linear(embed_dim, self.head_dim, bias=False),
                'W_v': nn.Linear(embed_dim, self.head_dim, bias=False),
            })
            for _ in range(num_heads)
        ])

        # Final projection: combines all head outputs
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = self.head_dim ** 0.5

    def forward(self, x):
        batch, seq_len, embed_dim = x.shape
        head_outputs = []

        for head in self.heads:
            Q = head['W_q'](x)   # (batch, seq, head_dim)
            K = head['W_k'](x)
            V = head['W_v'](x)

            scores = Q @ K.transpose(-2, -1) / self.scale
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
            scores = scores.masked_fill(~mask, float('-inf'))
            weights = torch.softmax(scores, dim=-1)

            head_outputs.append(weights @ V)  # (batch, seq, head_dim)

        # Concatenate all heads along the last dimension
        # [(batch, seq, 64), (batch, seq, 64), ...] → (batch, seq, 256)
        concat = torch.cat(head_outputs, dim=-1)
        return self.out_proj(concat)


embed_dim = 256
num_heads = 4
batch_size = 2
seq_len = 6

mha_naive = MultiHeadAttentionNaive(embed_dim, num_heads)
x = torch.randn(batch_size, seq_len, embed_dim)

output = mha_naive(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}  ← same shape preserved\n")

params = sum(p.numel() for p in mha_naive.parameters())
print(f"Parameters: {params:,}")
print(f"  4 heads × 3 matrices × (256×64) = {4 * 3 * 256 * 64:,}")
print(f"  + output projection (256×256)    = {256 * 256:,}")
print(f"  Total: {4 * 3 * 256 * 64 + 256 * 256:,}\n")


# ---------------------------------------------------------------
# STEP 3: EFFICIENT IMPLEMENTATION (tensor reshape trick)
# ---------------------------------------------------------------
print("=" * 50)
print("STEP 3: EFFICIENT MULTI-HEAD (production style)")
print("=" * 50)
print("""
The naive loop is slow because GPUs are optimized for large matrix ops,
not many small ones. The trick: do ALL heads in one big matrix multiply,
then reshape to split into heads.

Instead of 4 separate (256→64) projections,
do ONE (256→256) projection and reshape into (4, 64).

This is what real transformers (including GPT-2) actually do.
""")

class MultiHeadAttention(nn.Module):
    """
    Efficient multi-head attention using the reshape trick.
    This is the production-quality version used in real LLMs.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5

        # Single large projection for all heads at once
        # Instead of num_heads × (embed_dim, head_dim),
        # we use one (embed_dim, 3 * embed_dim) and split later
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq_len, embed_dim = x.shape

        # Project to Q, K, V — each shape: (batch, seq, embed_dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape to split embed_dim into (num_heads, head_dim)
        # (batch, seq, embed_dim) → (batch, seq, num_heads, head_dim)
        # → (batch, num_heads, seq, head_dim)  [transpose for matmul]
        def split_heads(t):
            return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(Q)  # (batch, num_heads, seq, head_dim)
        K = split_heads(K)
        V = split_heads(V)

        # Attention scores for ALL heads simultaneously
        scores = Q @ K.transpose(-2, -1) / self.scale
        # shape: (batch, num_heads, seq, seq)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Weighted sum of values
        # (batch, num_heads, seq, seq) @ (batch, num_heads, seq, head_dim)
        # → (batch, num_heads, seq, head_dim)
        out = weights @ V

        # Merge heads back: (batch, num_heads, seq, head_dim)
        # → (batch, seq, num_heads, head_dim) → (batch, seq, embed_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)

        return self.out_proj(out)


mha = MultiHeadAttention(embed_dim=256, num_heads=4)
output = mha(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}\n")

params = sum(p.numel() for p in mha.parameters())
print(f"Parameters: {params:,}  ← same as naive version, just faster\n")


# ---------------------------------------------------------------
# STEP 4: VISUALIZING WHAT EACH HEAD ATTENDS TO
# ---------------------------------------------------------------
print("=" * 50)
print("STEP 4: ATTENTION PATTERNS PER HEAD")
print("=" * 50)

# Peek inside: what are the attention weights for each head?
class MultiHeadAttentionInspectable(MultiHeadAttention):
    def forward(self, x):
        batch, seq_len, _ = x.shape
        Q = split_heads_fn(self.W_q(x), batch, seq_len, self.num_heads, self.head_dim)
        K = split_heads_fn(self.W_k(x), batch, seq_len, self.num_heads, self.head_dim)
        V = split_heads_fn(self.W_v(x), batch, seq_len, self.num_heads, self.head_dim)

        scores = Q @ K.transpose(-2, -1) / self.scale
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)

        out = weights @ V
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out), weights  # return weights too

def split_heads_fn(t, batch, seq_len, num_heads, head_dim):
    return t.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

mha_inspect = MultiHeadAttentionInspectable(embed_dim=256, num_heads=4)
output, attn_weights = mha_inspect(x)

# attn_weights shape: (batch, num_heads, seq, seq)
print(f"Attention weights shape: {attn_weights.shape}")
print(f"  ← (batch=2, num_heads=4, seq=6, seq=6)\n")

print("Attention pattern for batch 0, each head (6×6 matrix, rows=query, cols=key):")
for h in range(4):
    w = attn_weights[0, h].detach().numpy().round(2)
    print(f"\n  Head {h}:")
    for row in w:
        print(f"    {row}")

print("""
Each head has learned a different way to distribute attention.
In a trained model, these would correspond to meaningful patterns.
With random weights (untrained), they're just different random distributions.
""")


# ---------------------------------------------------------------
# STEP 5: SUMMARY + WHAT'S NEXT
# ---------------------------------------------------------------
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"""
Multi-head attention = run {num_heads} attention heads in parallel.

Each head:
  - Gets head_dim={embed_dim//num_heads} dimensions (a slice of embed_dim={embed_dim})
  - Independently computes Q, K, V and attention weights
  - Produces a (batch, seq, {embed_dim//num_heads}) output

All heads concatenated → (batch, seq, {embed_dim})
Final linear projection → (batch, seq, {embed_dim})

Key insight: same compute as one full head, but {num_heads}× more expressive
because each head can specialize in different relationship types.

This is the last building block. Next up: Stage 5 — the full GPT model.
We'll wrap multi-head attention in a Transformer Block (with LayerNorm
and a feedforward network), then stack multiple blocks to build GPT.
""")
