"""
Stage 3: Self-Attention
-----------------------
The core of the transformer. Every token looks at every other token
and decides how much to "attend" to it when building its representation.

The mechanism: Query, Key, Value
  - Query: "what am I looking for?"
  - Key:   "what do I contain?"
  - Value: "what do I share if attended to?"

Attention score = softmax(Q @ K.T / sqrt(head_dim)) @ V
"""

import torch
import torch.nn as nn
import tiktoken

torch.manual_seed(42)  # for reproducibility


# ---------------------------------------------------------------
# STEP 1: BUILD INTUITION WITH RAW MATH
# ---------------------------------------------------------------
print("=" * 50)
print("STEP 1: ATTENTION FROM SCRATCH (raw math)")
print("=" * 50)

# Tiny example: 3 tokens, each with embedding dim = 4
# (normally 768 or more, but 4 is easier to inspect)
x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # token 0
    [0.0, 1.0, 0.0, 1.0],   # token 1
    [1.0, 1.0, 0.0, 0.0],   # token 2
])
seq_len, embed_dim = x.shape
print(f"Input shape: {x.shape}  ← (3 tokens, 4-dim embeddings)\n")

# Create Q, K, V weight matrices (normally learned, here random)
head_dim = 4
W_q = torch.randn(embed_dim, head_dim)
W_k = torch.randn(embed_dim, head_dim)
W_v = torch.randn(embed_dim, head_dim)

# Project input into Q, K, V spaces
Q = x @ W_q   # shape: (3, 4)
K = x @ W_k   # shape: (3, 4)
V = x @ W_v   # shape: (3, 4)

print(f"Q (queries) shape: {Q.shape}")
print(f"K (keys) shape:    {K.shape}")
print(f"V (values) shape:  {V.shape}\n")

# Compute raw attention scores: Q @ K^T
# Result[i, j] = how much token i should attend to token j
scores = Q @ K.T
print(f"Raw attention scores shape: {scores.shape}  ← (each token vs each token)")
print(f"Raw scores:\n{scores.detach().numpy().round(2)}\n")

# Scale by sqrt(head_dim) — prevents scores from getting too large
# (large scores → extreme softmax → vanishing gradients)
scale = head_dim ** 0.5
scores = scores / scale
print(f"Scaled scores (÷ √{head_dim} = {scale:.2f}):\n{scores.detach().numpy().round(2)}\n")

# Softmax: convert scores to probabilities (each row sums to 1)
attn_weights = torch.softmax(scores, dim=-1)
print(f"Attention weights (after softmax):")
print(attn_weights.detach().numpy().round(3))
print(f"Row sums: {attn_weights.sum(dim=-1).tolist()}  ← each sums to 1\n")

# Final output: weighted sum of Values
output = attn_weights @ V
print(f"Output shape: {output.shape}  ← same as input, but context-aware")
print(f"Output:\n{output.detach().numpy().round(3)}\n")


# ---------------------------------------------------------------
# STEP 2: CAUSAL (MASKED) ATTENTION
# ---------------------------------------------------------------
print("=" * 50)
print("STEP 2: CAUSAL ATTENTION (masking future tokens)")
print("=" * 50)

# For language modeling, token at position i should ONLY attend
# to tokens at positions 0..i. It cannot see the future.
# (During inference, the future doesn't exist yet.)
# We enforce this with a causal mask.

print("Problem: token 0 attending to token 2 would be 'cheating'")
print("We apply a mask to prevent this:\n")

# Create a lower-triangular mask (True where we KEEP the score)
mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
print(f"Causal mask:\n{mask.int().numpy()}")
print("  1 = allowed to attend, 0 = masked out\n")

# Set masked positions to -infinity before softmax
# (softmax(-inf) = 0, so those positions get 0 attention weight)
masked_scores = scores.masked_fill(~mask, float('-inf'))
print(f"Scores after masking:\n{masked_scores.detach().numpy().round(2)}")
print("  (-inf values become 0 after softmax)\n")

causal_weights = torch.softmax(masked_scores, dim=-1)
print(f"Causal attention weights:")
print(causal_weights.detach().numpy().round(3))
print(f"  Token 0 only attends to itself")
print(f"  Token 1 attends to tokens 0-1")
print(f"  Token 2 attends to tokens 0-2\n")


# ---------------------------------------------------------------
# STEP 3: CLEAN SELF-ATTENTION MODULE
# ---------------------------------------------------------------
print("=" * 50)
print("STEP 3: SELF-ATTENTION AS A PYTORCH MODULE")
print("=" * 50)

class SelfAttention(nn.Module):
    """
    A single self-attention head.

    Takes a sequence of token embeddings and returns a new sequence
    where each token has gathered context from all previous tokens.
    """

    def __init__(self, embed_dim, head_dim):
        super().__init__()
        # Linear layers project input to Q, K, V spaces
        # bias=False is common in modern transformers
        self.W_q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, head_dim, bias=False)
        self.scale = head_dim ** 0.5

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        Q = self.W_q(x)   # (batch, seq_len, head_dim)
        K = self.W_k(x)   # (batch, seq_len, head_dim)
        V = self.W_v(x)   # (batch, seq_len, head_dim)

        # Attention scores
        # K.transpose(-2, -1) flips the last two dims for matrix multiply
        scores = Q @ K.transpose(-2, -1) / self.scale

        # Causal mask
        seq_len = x.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))

        # Attention weights
        weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        return weights @ V


# Test it with a realistic setup
embed_dim = 256
head_dim = 64
batch_size = 2
seq_len = 6

attn = SelfAttention(embed_dim, head_dim)

# Random input (normally this comes from the embedding layer)
x = torch.randn(batch_size, seq_len, embed_dim)
print(f"Input shape:  {x.shape}  ← (batch=2, seq=6, embed=256)")

output = attn(x)
print(f"Output shape: {output.shape}  ← same shape, now context-aware\n")

total_params = sum(p.numel() for p in attn.parameters())
print(f"Parameters in this attention head: {total_params:,}")
print(f"  (3 weight matrices of shape {embed_dim}×{head_dim} = {3 * embed_dim * head_dim:,})\n")


# ---------------------------------------------------------------
# STEP 4: WHAT IS THE MODEL ACTUALLY LEARNING?
# ---------------------------------------------------------------
print("=" * 50)
print("STEP 4: WHAT IS THE MODEL LEARNING?")
print("=" * 50)
print("""
The W_q, W_k, W_v matrices are the learnable parameters here.

During training, backpropagation adjusts these matrices so that
the attention weights become meaningful:

  - In "The animal didn't cross the street because IT was tired"
    → the attention weights for 'it' should be high for 'animal'

  - In "The bank by the river" vs "The bank approved the loan"
    → 'bank' should attend to different words in each context

The model figures this out entirely from data, with no rules given.

Key insight: attention is DYNAMIC — the weights change for every
input. Unlike a fixed lookup table, attention computes relationships
on the fly based on the actual content of the sequence.

Next up: Multi-Head Attention — instead of one attention head,
we run several in parallel, each learning different relationship types.
""")
