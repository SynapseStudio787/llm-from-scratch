"""
Stage 5: GPT Model Architecture
---------------------------------
We assemble all previous stages into a complete GPT model.

Architecture:
  Input IDs
  → Token Embedding + Position Embedding
  → N × TransformerBlock:
      LayerNorm → MultiHeadAttention → Residual
      LayerNorm → FeedForward (MLP)  → Residual
  → Final LayerNorm
  → Linear projection to vocab size
  → Logits (probability distribution over next token)
"""

import torch
import torch.nn as nn
import tiktoken

torch.manual_seed(42)


# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
# GPT-2 Small config (we'll use a smaller version for demo)
GPT2_SMALL = {
    "vocab_size":    50257,
    "context_length": 1024,
    "embed_dim":      768,
    "num_heads":       12,
    "num_layers":      12,
    "dropout":        0.1,
}

# Our smaller demo config (same structure, fewer params)
DEMO_CONFIG = {
    "vocab_size":    50257,
    "context_length": 256,
    "embed_dim":      256,
    "num_heads":        4,
    "num_layers":       4,
    "dropout":        0.1,
}

cfg = DEMO_CONFIG


# ---------------------------------------------------------------
# BUILDING BLOCK 1: Multi-Head Attention (from Stage 4)
# ---------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["embed_dim"] % cfg["num_heads"] == 0

        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["embed_dim"] // cfg["num_heads"]
        self.scale = self.head_dim ** 0.5

        self.W_q = nn.Linear(cfg["embed_dim"], cfg["embed_dim"], bias=False)
        self.W_k = nn.Linear(cfg["embed_dim"], cfg["embed_dim"], bias=False)
        self.W_v = nn.Linear(cfg["embed_dim"], cfg["embed_dim"], bias=False)
        self.out_proj = nn.Linear(cfg["embed_dim"], cfg["embed_dim"], bias=False)
        self.dropout = nn.Dropout(cfg["dropout"])

        # Pre-compute causal mask up to max context length
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg["context_length"], cfg["context_length"]))
        )

    def forward(self, x):
        batch, seq_len, embed_dim = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads: (batch, seq, embed) → (batch, heads, seq, head_dim)
        def split(t):
            return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split(Q), split(K), split(V)

        scores = Q @ K.transpose(-2, -1) / self.scale
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len] == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ V
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------
# BUILDING BLOCK 2: Feed-Forward Network (MLP)
# ---------------------------------------------------------------
print("=" * 50)
print("BUILDING BLOCK: FeedForward Network")
print("=" * 50)
print("""
After attention lets tokens COMMUNICATE with each other,
the feedforward network processes each token INDEPENDENTLY.

It's a simple 2-layer MLP:
  embed_dim → 4×embed_dim → embed_dim

The 4× expansion gives the model extra capacity to store
factual associations learned during training.

GELU activation (smoother than ReLU) is used between layers.
""")

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Applied independently to each token (same weights, different inputs).
    """
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["embed_dim"], cfg["embed_dim"]),
            nn.Dropout(cfg["dropout"]),
        )

    def forward(self, x):
        return self.net(x)

ff = FeedForward(cfg)
x_test = torch.randn(2, 6, cfg["embed_dim"])
print(f"FeedForward input:  {x_test.shape}")
print(f"FeedForward output: {ff(x_test).shape}  ← same shape\n")
params_ff = sum(p.numel() for p in ff.parameters())
print(f"Parameters: {params_ff:,}  (two linear layers with 4× expansion)\n")


# ---------------------------------------------------------------
# BUILDING BLOCK 3: Transformer Block
# ---------------------------------------------------------------
print("=" * 50)
print("BUILDING BLOCK: Transformer Block")
print("=" * 50)
print("""
One transformer block = attention + feedforward, each wrapped in:
  1. LayerNorm (applied BEFORE the sublayer — "pre-norm" style)
  2. Residual connection (add input back after sublayer)

Residual connections (x = x + sublayer(x)) are critical:
  - Gradients can flow directly back through addition
  - Even if sublayer learns nothing, output = input (safe default)
  - Enables training very deep networks (100+ layers)
""")

class TransformerBlock(nn.Module):
    """
    One transformer block:
      x → LayerNorm → MultiHeadAttention → + x  (residual)
        → LayerNorm → FeedForward        → + x  (residual)
    """
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])

    def forward(self, x):
        # Attention sublayer with residual
        x = x + self.attn(self.norm1(x))
        # Feedforward sublayer with residual
        x = x + self.ff(self.norm2(x))
        return x

block = TransformerBlock(cfg)
x_test = torch.randn(2, 6, cfg["embed_dim"])
print(f"TransformerBlock input:  {x_test.shape}")
print(f"TransformerBlock output: {block(x_test).shape}  ← shape preserved\n")
params_block = sum(p.numel() for p in block.parameters())
print(f"Parameters per block: {params_block:,}\n")


# ---------------------------------------------------------------
# THE FULL GPT MODEL
# ---------------------------------------------------------------
print("=" * 50)
print("THE FULL GPT MODEL")
print("=" * 50)

class GPT(nn.Module):
    """
    A GPT-style language model.

    Given a sequence of token IDs, predicts the probability
    distribution over the next token at each position.
    """
    def __init__(self, cfg):
        super().__init__()

        # Stage 2: Embeddings
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.pos_emb   = nn.Embedding(cfg["context_length"], cfg["embed_dim"])
        self.emb_drop  = nn.Dropout(cfg["dropout"])

        # Stage 4-5: Stack of transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(cfg["embed_dim"])

        # Output head: project from embed_dim → vocab_size
        # This gives us a score (logit) for each possible next token
        self.lm_head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)

        # Weight tying: share weights between token embedding and output head
        # The intuition: the embedding that represents a token as INPUT
        # should be related to recognizing that token as OUTPUT
        # This also saves ~13M parameters
        self.lm_head.weight = self.token_emb.weight

    def forward(self, token_ids):
        # token_ids shape: (batch, seq_len)
        batch, seq_len = token_ids.shape

        # Embeddings (Stage 2)
        tok = self.token_emb(token_ids)                         # (batch, seq, embed)
        pos = self.pos_emb(torch.arange(seq_len, device=token_ids.device))  # (seq, embed)
        x = self.emb_drop(tok + pos)

        # Transformer blocks (Stages 3+4)
        x = self.blocks(x)                                      # (batch, seq, embed)

        # Final norm + project to vocab
        x = self.final_norm(x)
        logits = self.lm_head(x)                                # (batch, seq, vocab_size)

        return logits

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        # Subtract tied weights (counted twice)
        tied = self.token_emb.weight.numel()
        return total - tied


# Instantiate the model
model = GPT(cfg)
print(f"Model parameters: {model.count_params():,}\n")

# Compare with GPT-2 sizes:
print("For reference:")
print(f"  Our demo model:  {model.count_params():>12,} params")
print(f"  GPT-2 Small:     {117_000_000:>12,} params  (12 layers, embed=768)")
print(f"  GPT-2 Large:     {774_000_000:>12,} params  (36 layers, embed=1280)")
print(f"  GPT-3:           {175_000_000_000:>12,} params  (96 layers, embed=12288)")
print()


# ---------------------------------------------------------------
# FORWARD PASS: token IDs → logits
# ---------------------------------------------------------------
print("=" * 50)
print("FORWARD PASS")
print("=" * 50)

tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, I am"
token_ids = tokenizer.encode(text)
input_tensor = torch.tensor(token_ids).unsqueeze(0)  # add batch dim

print(f"Input text:    '{text}'")
print(f"Token IDs:     {token_ids}")
print(f"Input shape:   {input_tensor.shape}  ← (batch=1, seq={len(token_ids)})\n")

model.eval()
with torch.no_grad():
    logits = model(input_tensor)

print(f"Logits shape: {logits.shape}  ← (batch=1, seq={len(token_ids)}, vocab=50257)")
print(f"""
For each of the {len(token_ids)} positions, the model outputs 50,257 scores.
The score at position i represents: "given tokens 0..i, what's the next token?"

We care most about the LAST position — that's predicting what comes after '{tokenizer.decode([token_ids[-1]])}'.
""")

last_logits = logits[0, -1, :]  # shape: (50257,)
print(f"Logits at last position (first 5): {last_logits[:5].tolist()}")
print(f"(random/meaningless — model is untrained)\n")

# Convert logits to probabilities
probs = torch.softmax(last_logits, dim=-1)
top5 = torch.topk(probs, 5)
print("Top 5 predicted next tokens (untrained model = random):")
for prob, idx in zip(top5.values, top5.indices):
    token_str = tokenizer.decode([idx.item()])
    print(f"  '{token_str}' → {prob.item():.4f}")


# ---------------------------------------------------------------
# TEXT GENERATION (greedy)
# ---------------------------------------------------------------
print()
print("=" * 50)
print("TEXT GENERATION (greedy decoding)")
print("=" * 50)
print("""
Generation loop:
  1. Feed current token sequence into model
  2. Take logits at the last position
  3. Pick the highest-scoring token (greedy)
  4. Append it to the sequence
  5. Repeat
""")

def generate(model, tokenizer, prompt, max_new_tokens=10):
    model.eval()
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(ids).unsqueeze(0)

    print(f"Prompt: '{prompt}'")
    print(f"Generating {max_new_tokens} tokens...\n")

    with torch.no_grad():
        for step in range(max_new_tokens):
            logits = model(input_ids)
            next_logits = logits[0, -1, :]           # last position
            next_id = torch.argmax(next_logits).item()  # greedy: pick best
            next_token = tokenizer.decode([next_id])

            print(f"  Step {step+1}: predicted '{next_token}' (ID={next_id})")
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_id]])
            ], dim=1)

    generated = tokenizer.decode(input_ids[0].tolist())
    print(f"\nFull output: '{generated}'")
    print("(Gibberish — model has random weights. Training fixes this.)")

generate(model, tokenizer, "Hello, I am", max_new_tokens=8)


# ---------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------
print()
print("=" * 50)
print("SUMMARY: What we built")
print("=" * 50)
print(f"""
GPT Model structure (demo config):
  Token embedding:     50,257 × 256
  Position embedding:  256 × 256
  {cfg['num_layers']} × TransformerBlock:
    MultiHeadAttention ({cfg['num_heads']} heads × {cfg['embed_dim']//cfg['num_heads']} dim)
    FeedForward (256 → 1024 → 256)
    2 × LayerNorm
  Final LayerNorm
  Output projection:   256 × 50,257 (tied with token embedding)

The model is architecturally complete. It can run a forward pass
and generate text — just nonsense text because weights are random.

Next up: Stage 6 — Pretraining.
We give the model real text data and train it to predict the next token.
That's what turns random weights into a language model.
""")
