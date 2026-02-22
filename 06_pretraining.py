"""
Stage 6: Pretraining
---------------------
We train the GPT model on real text using next-token prediction.

The objective: given tokens [t0, t1, ..., tn], predict [t1, t2, ..., tn+1]
Loss = cross-entropy between predicted logits and actual next tokens
Optimizer = AdamW (Adam with weight decay regularization)

We use a small text corpus and our demo model so training runs on CPU
in a reasonable time. The principle is identical to GPT-2/3 pretraining —
just at a fraction of the scale.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

torch.manual_seed(42)


# ---------------------------------------------------------------
# THE FULL MODEL (copied from Stage 5 — self-contained)
# ---------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["embed_dim"] // cfg["num_heads"]
        self.scale = self.head_dim ** 0.5
        self.W_q = nn.Linear(cfg["embed_dim"], cfg["embed_dim"], bias=False)
        self.W_k = nn.Linear(cfg["embed_dim"], cfg["embed_dim"], bias=False)
        self.W_v = nn.Linear(cfg["embed_dim"], cfg["embed_dim"], bias=False)
        self.out_proj = nn.Linear(cfg["embed_dim"], cfg["embed_dim"], bias=False)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.register_buffer("mask", torch.tril(torch.ones(cfg["context_length"], cfg["context_length"])))

    def forward(self, x):
        B, T, C = x.shape
        def split(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        Q, K, V = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        scores = (Q @ K.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["embed_dim"], cfg["embed_dim"]),
            nn.Dropout(cfg["dropout"]),
        )
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.pos_emb   = nn.Embedding(cfg["context_length"], cfg["embed_dim"])
        self.emb_drop  = nn.Dropout(cfg["dropout"])
        self.blocks    = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["num_layers"])])
        self.final_norm = nn.LayerNorm(cfg["embed_dim"])
        self.lm_head   = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, token_ids):
        B, T = token_ids.shape
        x = self.emb_drop(self.token_emb(token_ids) + self.pos_emb(torch.arange(T, device=token_ids.device)))
        x = self.final_norm(self.blocks(x))
        return self.lm_head(x)


# ---------------------------------------------------------------
# CONFIG — extra small for fast CPU training
# ---------------------------------------------------------------
cfg = {
    "vocab_size":     50257,
    "context_length":   128,
    "embed_dim":        128,
    "num_heads":          4,
    "num_layers":         2,
    "dropout":          0.0,   # 0 for small datasets
}


# ---------------------------------------------------------------
# TRAINING TEXT
# ---------------------------------------------------------------
# A short public-domain excerpt (enough tokens to train on)
RAW_TEXT = """
The quick brown fox jumps over the lazy dog.
To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune,
or to take arms against a sea of troubles and by opposing end them.
All the world's a stage, and all the men and women merely players.
They have their exits and their entrances, and one man in his time plays many parts.
It was the best of times, it was the worst of times, it was the age of wisdom,
it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity.
We hold these truths to be self-evident, that all men are created equal.
In the beginning God created the heavens and the earth.
To infinity and beyond. The only limit to our realization of tomorrow is our doubts of today.
Ask not what your country can do for you, ask what you can do for your country.
The greatest glory in living lies not in never falling, but in rising every time we fall.
In the end it is not the years in your life that count, it is the life in your years.
The journey of a thousand miles begins with one step. Life is what happens when you are busy making other plans.
Spread love everywhere you go. When you reach the end of your rope, tie a knot in it and hang on.
Always remember that you are absolutely unique, just like everyone else.
Do not go where the path may lead, go instead where there is no path and leave a trail.
You will face many defeats in life, but never let yourself be defeated.
The most difficult thing is the decision to act, the rest is merely tenacity.
How wonderful it is that nobody need wait a single moment before starting to improve the world.
When I was five years old, my mother told me that happiness was the key to life.
Spread your wings and fly away into the horizon where the sky meets the earth.
Knowledge is power and power is knowledge and knowing what you know is half the battle.
The mind is everything. What you think you become. What you feel you attract.
To the mind that is still, the whole universe surrenders and reveals its secrets.
Life is not measured by the number of breaths we take but by the moments that take our breath away.
""" * 6   # repeat to get more training data


# ---------------------------------------------------------------
# DATASET: sliding window over tokenized text
# ---------------------------------------------------------------
print("=" * 50)
print("STEP 1: DATASET CREATION")
print("=" * 50)
print("""
We slide a window of context_length tokens across the full text.
For each window:
  input  = tokens[i   : i+context_length]
  target = tokens[i+1 : i+context_length+1]  ← shifted by 1

The model sees 'input' and learns to predict 'target'.
Every position in the window contributes to the loss.
""")

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, context_length, stride):
        self.inputs  = []
        self.targets = []

        token_ids = tokenizer.encode(text)
        print(f"Total tokens in training text: {len(token_ids):,}")

        # Slide window with given stride (stride < context_length = overlapping windows)
        for i in range(0, len(token_ids) - context_length, stride):
            inp = token_ids[i            : i + context_length]
            tgt = token_ids[i + 1        : i + context_length + 1]
            self.inputs.append(torch.tensor(inp))
            self.targets.append(torch.tensor(tgt))

        print(f"Training samples created: {len(self.inputs)}")
        print(f"Each sample: {context_length} input tokens → {context_length} target tokens\n")

    def __len__(self):  return len(self.inputs)
    def __getitem__(self, i): return self.inputs[i], self.targets[i]


tokenizer = tiktoken.get_encoding("gpt2")
context_length = cfg["context_length"]

dataset = TextDataset(RAW_TEXT, tokenizer, context_length=context_length, stride=context_length // 2)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Show one batch
batch_inputs, batch_targets = next(iter(dataloader))
print(f"Batch input shape:  {batch_inputs.shape}   ← (batch=4, seq=128)")
print(f"Batch target shape: {batch_targets.shape}   ← (batch=4, seq=128)")
print()
print("First sample — input vs target (first 10 tokens):")
sample_in  = batch_inputs[0, :10].tolist()
sample_tgt = batch_targets[0, :10].tolist()
print(f"  Input:  {[tokenizer.decode([t]) for t in sample_in]}")
print(f"  Target: {[tokenizer.decode([t]) for t in sample_tgt]}")
print("  ← each target is the next token after the corresponding input\n")


# ---------------------------------------------------------------
# LOSS FUNCTION
# ---------------------------------------------------------------
print("=" * 50)
print("STEP 2: THE LOSS FUNCTION")
print("=" * 50)
print("""
Cross-entropy loss measures how wrong the model's predictions are.

For each position, the model gives 50,257 logit scores.
We look at the logit for the ACTUAL next token.
A perfect model would give that token the highest score → low loss.
A random model distributes scores evenly → high loss ≈ ln(50257) ≈ 10.8

Perplexity = exp(loss). A perplexity of 10 means the model is
roughly as confused as if it were choosing uniformly among 10 tokens.
""")

def compute_loss(model, inputs, targets):
    logits = model(inputs)    # (batch, seq, vocab_size)
    # Flatten to (batch*seq, vocab_size) and (batch*seq,) for cross-entropy
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )
    return loss


# ---------------------------------------------------------------
# TEXT GENERATION (for before/after comparison)
# ---------------------------------------------------------------
def generate(model, tokenizer, prompt, max_new_tokens=30, temperature=1.0):
    model.eval()
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(ids).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to context length
            input_ids = input_ids[:, -cfg["context_length"]:]
            logits = model(input_ids)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            # Sample from distribution (more interesting than pure greedy)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


# ---------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------
print("=" * 50)
print("STEP 3: TRAINING")
print("=" * 50)

model = GPT(cfg)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# AdamW: Adam optimizer + weight decay
# Weight decay adds a small penalty for large weights (regularization)
# lr=5e-3 is relatively high — fine for small datasets, speeds up demo
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.1)

print("\n--- Before training ---")
with torch.no_grad():
    sample_in, sample_tgt = dataset[0]
    initial_loss = compute_loss(model, sample_in.unsqueeze(0), sample_tgt.unsqueeze(0))
print(f"Initial loss: {initial_loss.item():.4f}  (random ≈ {torch.log(torch.tensor(cfg['vocab_size'])).item():.2f})")
print(f"Initial perplexity: {torch.exp(initial_loss).item():.1f}")
print()
print(f"Sample generation before training:")
print(f"  '{generate(model, tokenizer, 'The quick brown', max_new_tokens=20)}'")
print()

# Training loop
num_epochs = 15
print(f"Training for {num_epochs} epochs...\n")
print(f"{'Epoch':>6} {'Step':>6} {'Loss':>8} {'Perplexity':>12}")
print("-" * 38)

losses = []
step = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for inputs, targets in dataloader:
        optimizer.zero_grad()           # clear old gradients
        loss = compute_loss(model, inputs, targets)
        loss.backward()                 # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent gradient explosion
        optimizer.step()               # update weights

        epoch_loss += loss.item()
        num_batches += 1
        step += 1

    avg_loss = epoch_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    losses.append(avg_loss)

    if (epoch + 1) % 3 == 0 or epoch == 0:
        print(f"{epoch+1:>6} {step:>6} {avg_loss:>8.4f} {perplexity:>12.1f}")

print()


# ---------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------
print("=" * 50)
print("STEP 4: RESULTS")
print("=" * 50)

model.eval()
with torch.no_grad():
    final_loss = compute_loss(model, sample_in.unsqueeze(0), sample_tgt.unsqueeze(0))

print(f"Initial loss:  {initial_loss.item():.4f}  → perplexity {torch.exp(initial_loss).item():.1f}")
print(f"Final loss:    {final_loss.item():.4f}  → perplexity {torch.exp(final_loss).item():.1f}")
print(f"Improvement:   {((initial_loss - final_loss) / initial_loss * 100).item():.1f}% loss reduction\n")

print("Sample generation after training:")
prompts = [
    "The quick brown",
    "To be or not",
    "It was the best",
    "In the beginning",
]
for prompt in prompts:
    result = generate(model, tokenizer, prompt, max_new_tokens=20, temperature=0.8)
    print(f"  '{result}'")

print("""
The model has learned patterns from this small corpus.
With a larger dataset (billions of tokens) and larger model,
these patterns become grammar, facts, reasoning, and style.

Next up: Stage 7 — Fine-tuning.
We take a pretrained model and specialize it for a specific task
(e.g., following instructions, answering questions, classification).
""")


# ---------------------------------------------------------------
# WHAT BACKPROPAGATION ACTUALLY DOES
# ---------------------------------------------------------------
print("=" * 50)
print("HOW TRAINING WORKS UNDER THE HOOD")
print("=" * 50)
print("""
loss.backward() computes the gradient of the loss with respect
to every single parameter in the model (millions of numbers).

The gradient tells us: "if we increase this weight slightly,
does the loss go up or down, and by how much?"

optimizer.step() nudges each weight slightly in the direction
that decreases the loss (gradient descent).

After thousands of steps, the weights settle into values where
the model has learned the statistical patterns of the training text.

Gradient clipping (clip_grad_norm_) prevents any single update
from being so large it destabilizes training — common in transformers.
""")
