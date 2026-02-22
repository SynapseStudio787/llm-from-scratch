"""
Stage 2: Embeddings
-------------------
Token IDs are just arbitrary integers — they carry no meaning.
We need to convert them into vectors of floating point numbers
that the model can do math with, and that capture semantic relationships.

Two embeddings are added together:
  1. Token embeddings  — what the word means
  2. Position embeddings — where in the sequence it appears
"""

import torch
import tiktoken

# ---------------------------------------------------------------
# PYTORCH PRIMER: what is a tensor?
# ---------------------------------------------------------------
# PyTorch's core data structure is a tensor — basically a multi-
# dimensional array of numbers, like numpy arrays but with built-in
# support for GPU acceleration and automatic gradient computation.
#
#   1D tensor (vector):  [1.0, 2.0, 3.0]
#   2D tensor (matrix):  [[1, 2], [3, 4]]
#   3D tensor:           a stack of matrices
#
# Shape notation: (batch_size, sequence_length, embedding_dim)
# We'll see this pattern constantly.

print("=" * 50)
print("PYTORCH PRIMER: Tensors")
print("=" * 50)

# A simple 1D tensor
t = torch.tensor([1.0, 2.0, 3.0])
print(f"1D tensor: {t}")
print(f"Shape: {t.shape}\n")

# A 2D tensor (matrix)
m = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print(f"2D tensor:\n{m}")
print(f"Shape: {m.shape}  ← (3 rows, 2 columns)\n")


# ---------------------------------------------------------------
# SETUP: tokenize a sample text
# ---------------------------------------------------------------
print("=" * 50)
print("SETUP: Tokenizing sample text")
print("=" * 50)

tokenizer = tiktoken.get_encoding("gpt2")
text = "The cat sat on the mat"
token_ids = tokenizer.encode(text)

print(f"Text:      '{text}'")
print(f"Token IDs: {token_ids}")
print(f"Tokens:    {[tokenizer.decode([i]) for i in token_ids]}\n")

# Convert to a PyTorch tensor so the model can process it
input_ids = torch.tensor(token_ids)
print(f"As tensor: {input_ids}")
print(f"Shape: {input_ids.shape}  ← ({len(token_ids)} tokens)\n")


# ---------------------------------------------------------------
# 1. TOKEN EMBEDDINGS
# ---------------------------------------------------------------
print("=" * 50)
print("TOKEN EMBEDDINGS")
print("=" * 50)

# GPT-2 configuration
vocab_size = 50257   # number of unique tokens in the vocabulary
embedding_dim = 256  # size of each embedding vector
                     # (GPT-2 uses 768, we use 256 for demo)

# nn.Embedding is a lookup table:
#   - vocab_size rows (one per token)
#   - embedding_dim columns (the vector for that token)
# Initially filled with random numbers — training adjusts these.
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

print(f"Embedding table shape: {token_embedding_layer.weight.shape}")
print(f"  ← {vocab_size:,} tokens × {embedding_dim} dimensions each\n")

# Look up embeddings for our token IDs
token_embeddings = token_embedding_layer(input_ids)

print(f"Input IDs shape:       {input_ids.shape}          ← 6 token IDs")
print(f"Token embeddings shape: {token_embeddings.shape}  ← 6 tokens, each a {embedding_dim}-dim vector\n")

print("First token's embedding vector (first 8 values):")
print(f"  {token_embeddings[0, :8].tolist()}")
print("  (random for now — training will give these values meaning)\n")


# ---------------------------------------------------------------
# 2. POSITION EMBEDDINGS
# ---------------------------------------------------------------
print("=" * 50)
print("POSITION EMBEDDINGS")
print("=" * 50)

# Why do we need this?
# Self-attention (coming next) looks at ALL tokens simultaneously.
# Without position info, "cat sat on mat" and "mat on sat cat"
# would produce identical representations. Order would be invisible.
#
# Solution: add a second embedding that encodes position (0, 1, 2, ...)
# The model learns what "being in position 0" vs "position 5" means.

context_length = len(token_ids)  # how many tokens in our sequence
position_embedding_layer = torch.nn.Embedding(context_length, embedding_dim)

# Create position indices: [0, 1, 2, 3, 4, 5]
positions = torch.arange(context_length)
print(f"Position indices: {positions.tolist()}")

position_embeddings = position_embedding_layer(positions)
print(f"Position embeddings shape: {position_embeddings.shape}  ← same shape as token embeddings\n")


# ---------------------------------------------------------------
# 3. COMBINING THEM: token + position
# ---------------------------------------------------------------
print("=" * 50)
print("COMBINED EMBEDDINGS")
print("=" * 50)

# Simply add them element-wise.
# Each token's final representation = what it IS + where it IS.
input_embeddings = token_embeddings + position_embeddings

print(f"Token embeddings shape:    {token_embeddings.shape}")
print(f"Position embeddings shape: {position_embeddings.shape}")
print(f"Combined shape:            {input_embeddings.shape}")
print()
print("These combined vectors are what gets fed into the transformer.")
print("Shape is (sequence_length=6, embedding_dim=256)")
print()


# ---------------------------------------------------------------
# 4. BATCHING: processing multiple sequences at once
# ---------------------------------------------------------------
print("=" * 50)
print("BATCHING: processing multiple sequences")
print("=" * 50)

# In real training, we don't process one sentence at a time —
# we process a batch of many sequences in parallel for efficiency.
# This adds a batch dimension to our tensors.

text2 = "Dogs are loyal animals"
ids2 = tokenizer.encode(text2)

# Pad or truncate to same length for batching (simplified here)
ids1_padded = token_ids[:4]
ids2_padded = ids2[:4]

batch = torch.tensor([ids1_padded, ids2_padded])
print(f"Batch shape: {batch.shape}  ← (2 sequences, 4 tokens each)")

batch_embeddings = token_embedding_layer(batch)
print(f"Batch embeddings shape: {batch_embeddings.shape}")
print(f"  ← (batch_size=2, sequence_length=4, embedding_dim={embedding_dim})")
print()
print("This (batch, sequence, embedding) shape is the standard throughout")
print("the entire transformer architecture.\n")


# ---------------------------------------------------------------
# 5. SUMMARY
# ---------------------------------------------------------------
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print("""
Text → Token IDs (tokenizer)
Token IDs → Token Embeddings (lookup table, learned)
Positions → Position Embeddings (lookup table, learned)
Token Embeddings + Position Embeddings → Input to Transformer

The embedding dimension (256 here, 768 in GPT-2, 12288 in GPT-3)
is one of the key hyperparameters that determines model capacity.

Next up: Attention — how the model lets each token look at
and gather information from every other token in the sequence.
""")
