"""
Stage 7: Fine-tuning
---------------------
Pretraining gives us a model that understands language.
Fine-tuning specializes it for a specific task.

Two approaches demonstrated:
  1. Classification fine-tuning — add a classifier head, train on labels
  2. Instruction fine-tuning    — train on prompt/response pairs

Key insight: fine-tuning starts from pretrained weights.
The model already understands language; we just redirect it.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

torch.manual_seed(42)


# ---------------------------------------------------------------
# THE MODEL (same as Stage 5/6)
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
        def split(t): return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
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
        self.attn  = MultiHeadAttention(cfg)
        self.ff    = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["embed_dim"])
        self.norm2 = nn.LayerNorm(cfg["embed_dim"])
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb  = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.pos_emb    = nn.Embedding(cfg["context_length"], cfg["embed_dim"])
        self.emb_drop   = nn.Dropout(cfg["dropout"])
        self.blocks     = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["num_layers"])])
        self.final_norm = nn.LayerNorm(cfg["embed_dim"])
        self.lm_head    = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, ids):
        B, T = ids.shape
        x = self.emb_drop(self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device)))
        return self.lm_head(self.final_norm(self.blocks(x)))

    def get_hidden_states(self, ids):
        """Return final hidden states without the LM head — used for classification."""
        B, T = ids.shape
        x = self.emb_drop(self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device)))
        return self.final_norm(self.blocks(x))   # (B, T, embed_dim)


cfg = {
    "vocab_size":     50257,
    "context_length":   128,
    "embed_dim":        128,
    "num_heads":          4,
    "num_layers":         2,
    "dropout":          0.0,
}

tokenizer = tiktoken.get_encoding("gpt2")


# ===============================================================
# PART 1: CLASSIFICATION FINE-TUNING
# ===============================================================
print("=" * 55)
print("PART 1: CLASSIFICATION FINE-TUNING")
print("=" * 55)
print("""
Task: Sentiment analysis — is a review positive or negative?

Architecture change:
  Pretrained GPT  →  [Transformer Blocks]  →  LM head (vocab)
  Fine-tuned      →  [Transformer Blocks]  →  Classifier head (2 classes)

We take the LAST token's hidden state (it has attended to the whole
sequence) and pass it through a small linear layer → 2 logits.

The transformer blocks keep their pretrained weights as a starting
point. Only the new classifier head starts from random.
""")


# ---------------------------------------------------------------
# 1a. LABELED DATASET
# ---------------------------------------------------------------
positive_reviews = [
    "I absolutely loved this movie, it was fantastic!",
    "The food was delicious and the service was wonderful.",
    "Best purchase I have ever made, highly recommend.",
    "An incredible experience from start to finish.",
    "This product exceeded all my expectations.",
    "Brilliant storytelling and amazing performances.",
    "I had such a great time, will definitely return.",
    "Outstanding quality and arrived ahead of schedule.",
    "Five stars, simply the best I have ever tried.",
    "Loved every moment, truly a magical experience.",
    "The staff were so kind and helpful throughout.",
    "Perfect in every way, I am so happy with it.",
]

negative_reviews = [
    "Terrible product, broke after just two days.",
    "The worst experience I have ever had, avoid this.",
    "Completely disappointed, nothing like advertised.",
    "Horrible customer service, never buying again.",
    "Total waste of money, do not recommend at all.",
    "The food was cold and the waiter was rude.",
    "Fell apart immediately, absolute garbage quality.",
    "I want my money back, this is completely unacceptable.",
    "Worst movie I have seen, boring and poorly made.",
    "Failed to work from the start, very frustrating.",
    "Disgusting experience, the place was filthy.",
    "Nothing worked as described, a total scam.",
]

# Label: 1 = positive, 0 = negative
all_texts  = positive_reviews + negative_reviews
all_labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

print(f"Dataset: {len(positive_reviews)} positive + {len(negative_reviews)} negative reviews")
print(f"Example positive: '{positive_reviews[0]}'")
print(f"Example negative: '{negative_reviews[0]}'\n")


# ---------------------------------------------------------------
# 1b. DATASET CLASS
# ---------------------------------------------------------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.data = []
        for text, label in zip(texts, labels):
            ids = tokenizer.encode(text)[:max_length]
            # Pad to max_length so batches are uniform
            padded = ids + [tokenizer.eot_token] * (max_length - len(ids))
            self.data.append((torch.tensor(padded), torch.tensor(label)))

    def __len__(self):  return len(self.data)
    def __getitem__(self, i): return self.data[i]

max_length = 32
dataset = SentimentDataset(all_texts, all_labels, tokenizer, max_length)

# 80/20 train/val split
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=4)

print(f"Train: {train_size} samples | Val: {val_size} samples\n")


# ---------------------------------------------------------------
# 1c. CLASSIFIER MODEL
# ---------------------------------------------------------------
class GPTClassifier(nn.Module):
    """
    GPT backbone + a small classification head.

    The backbone's hidden states capture rich language understanding.
    The classifier head maps the last token's state → class logits.
    """
    def __init__(self, gpt_model, num_classes=2):
        super().__init__()
        self.gpt = gpt_model
        embed_dim = gpt_model.token_emb.embedding_dim
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, ids):
        # Get hidden states from backbone
        hidden = self.gpt.get_hidden_states(ids)   # (B, T, embed_dim)
        # Use the LAST token — it has attended to the full sequence
        last = hidden[:, -1, :]                    # (B, embed_dim)
        return self.classifier(last)               # (B, num_classes)


# Start from a fresh GPT (in real fine-tuning, you'd load pretrained weights)
base_model = GPT(cfg)
classifier = GPTClassifier(base_model, num_classes=2)

total_params   = sum(p.numel() for p in classifier.parameters())
backbone_params = sum(p.numel() for p in classifier.gpt.parameters())
head_params    = sum(p.numel() for p in classifier.classifier.parameters())

print(f"Classifier model parameters:")
print(f"  GPT backbone:      {backbone_params:>10,}")
print(f"  Classifier head:   {head_params:>10,}  ← only 258 new params!")
print(f"  Total:             {total_params:>10,}\n")


# ---------------------------------------------------------------
# 1d. TRAINING
# ---------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ids, labels in loader:
            logits = model(ids)
            preds  = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total

optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=0.01)

print("Training sentiment classifier...\n")
print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Acc':>10}")
print("-" * 44)

# Evaluate before training
val_acc_before = evaluate(classifier, val_loader)
print(f"{'0 (init)':>6} {'—':>12} {'—':>10} {val_acc_before:>9.1%}")

for epoch in range(12):
    classifier.train()
    total_loss = 0
    for ids, labels in train_loader:
        optimizer.zero_grad()
        logits = classifier(ids)
        loss   = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 3 == 0:
        train_acc = evaluate(classifier, train_loader)
        val_acc   = evaluate(classifier, val_loader)
        avg_loss  = total_loss / len(train_loader)
        print(f"{epoch+1:>6} {avg_loss:>12.4f} {train_acc:>9.1%} {val_acc:>10.1%}")

print()

# Test on new examples never seen during training
print("Testing on unseen examples:")
test_examples = [
    ("This was an absolutely wonderful meal.", 1),
    ("Complete disaster, nothing worked right.", 0),
    ("Really enjoyed every part of this trip.", 1),
    ("Worst thing I have ever bought.", 0),
]

classifier.eval()
with torch.no_grad():
    for text, true_label in test_examples:
        ids = tokenizer.encode(text)[:max_length]
        padded = ids + [tokenizer.eot_token] * (max_length - len(ids))
        input_tensor = torch.tensor(padded).unsqueeze(0)

        logits = classifier(input_tensor)
        probs  = torch.softmax(logits, dim=-1)
        pred   = logits.argmax(dim=-1).item()
        label_str = "POSITIVE" if pred == 1 else "NEGATIVE"
        correct   = "✓" if pred == true_label else "✗"

        print(f"  {correct} '{text}'")
        print(f"    → {label_str}  (pos={probs[0,1]:.2f}, neg={probs[0,0]:.2f})")


# ===============================================================
# PART 2: INSTRUCTION FINE-TUNING
# ===============================================================
print()
print("=" * 55)
print("PART 2: INSTRUCTION FINE-TUNING")
print("=" * 55)
print("""
This is what turns a pretrained GPT into an assistant.

The data format: each example is a (prompt, response) pair.
We format it as:

  <|prompt|>What is the capital of France?<|response|>Paris.<|endoftext|>

Critical difference from pretraining:
  - Pretraining:  compute loss on ALL tokens
  - Instruction:  compute loss ONLY on response tokens (masked)

Why? We don't want to "reward" the model for predicting the prompt
— the prompt is given. We only want it to learn to produce the response.
""")


# ---------------------------------------------------------------
# 2a. INSTRUCTION DATASET
# ---------------------------------------------------------------
instruction_pairs = [
    ("What is the capital of France?",        "Paris."),
    ("What is 2 plus 2?",                     "4."),
    ("Name a primary color.",                 "Red."),
    ("What planet do we live on?",            "Earth."),
    ("What is the opposite of hot?",          "Cold."),
    ("How many days in a week?",              "Seven."),
    ("What do bees make?",                    "Honey."),
    ("What is the largest ocean?",            "The Pacific Ocean."),
    ("How many sides does a triangle have?",  "Three."),
    ("What is the color of the sky?",         "Blue."),
    ("What animal says meow?",                "A cat."),
    ("What is the boiling point of water?",   "100 degrees Celsius."),
]

PROMPT_TOKEN  = "<|endoftext|>"   # reusing special token as separator
RESPONSE_TOKEN = "<|endoftext|>"

def format_example(prompt, response, tokenizer):
    """
    Format a prompt/response pair into token IDs.
    Returns (input_ids, loss_mask) where loss_mask marks response tokens.
    """
    prompt_text   = f"### Prompt:\n{prompt}\n\n### Response:\n"
    response_text = f"{response}"
    full_text     = prompt_text + response_text

    prompt_ids   = tokenizer.encode(prompt_text)
    response_ids = tokenizer.encode(response_text)
    full_ids     = prompt_ids + response_ids

    # Loss mask: 0 for prompt tokens (ignore), 1 for response tokens (learn)
    loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

    return full_ids, loss_mask


# Show the formatting
print("Example instruction format:")
example_ids, example_mask = format_example(
    instruction_pairs[0][0], instruction_pairs[0][1], tokenizer
)
print(f"  Prompt+Response tokens: {[tokenizer.decode([i]) for i in example_ids]}")
print(f"  Loss mask:              {example_mask}")
print(f"  ← 0 = ignore (prompt), 1 = learn from (response)\n")


# ---------------------------------------------------------------
# 2b. INSTRUCTION TRAINING LOOP
# ---------------------------------------------------------------
class InstructionDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=64):
        self.data = []
        for prompt, response in pairs:
            ids, mask = format_example(prompt, response, tokenizer)
            ids  = ids[:max_length]
            mask = mask[:max_length]
            pad  = max_length - len(ids)
            ids  += [tokenizer.eot_token] * pad
            mask += [0] * pad
            self.data.append((torch.tensor(ids), torch.tensor(mask)))

    def __len__(self):  return len(self.data)
    def __getitem__(self, i): return self.data[i]


def instruction_loss(model, ids, loss_mask):
    """
    Cross-entropy loss, but only on response tokens.
    Prompt tokens contribute 0 to the loss.
    """
    logits  = model(ids)                            # (B, T, vocab)
    B, T, V = logits.shape
    targets = ids[:, 1:]                            # shift right: predict next token
    logits  = logits[:, :-1, :]                     # align with targets
    mask    = loss_mask[:, 1:].float()              # shift mask too

    # Per-token cross-entropy
    loss_per_token = nn.functional.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        reduction='none'
    ).view(B, T - 1)

    # Zero out prompt tokens; average over response tokens only
    masked = (loss_per_token * mask).sum() / mask.sum().clamp(min=1)
    return masked


# Fresh model for instruction tuning
inst_model = GPT(cfg)
inst_optimizer = torch.optim.AdamW(inst_model.parameters(), lr=5e-3)

inst_dataset = InstructionDataset(instruction_pairs, tokenizer)
inst_loader  = DataLoader(inst_dataset, batch_size=4, shuffle=True)

print("Training instruction model...\n")
print(f"{'Epoch':>6} {'Loss':>10}")
print("-" * 18)

for epoch in range(20):
    inst_model.train()
    total = 0
    for ids, mask in inst_loader:
        inst_optimizer.zero_grad()
        loss = instruction_loss(inst_model, ids, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inst_model.parameters(), 1.0)
        inst_optimizer.step()
        total += loss.item()
    avg = total / len(inst_loader)
    if (epoch + 1) % 5 == 0:
        print(f"{epoch+1:>6} {avg:>10.4f}")

print()


# ---------------------------------------------------------------
# 2c. GENERATION WITH INSTRUCTION MODEL
# ---------------------------------------------------------------
def generate_response(model, tokenizer, prompt, max_new=20):
    model.eval()
    prompt_text = f"### Prompt:\n{prompt}\n\n### Response:\n"
    ids = tokenizer.encode(prompt_text)
    input_ids = torch.tensor(ids).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new):
            input_ids = input_ids[:, -cfg["context_length"]:]
            logits    = model(input_ids)
            next_id   = logits[0, -1, :].argmax().item()
            if next_id == tokenizer.eot_token:
                break
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)

    # Return only the generated response (after the prompt)
    full = tokenizer.decode(input_ids[0].tolist())
    return full.split("### Response:\n")[-1].strip()


print("Instruction model responses (greedy):")
test_prompts = [
    "What is the capital of France?",
    "How many sides does a triangle have?",
    "What animal says meow?",
    "What is the opposite of hot?",
]
for p in test_prompts:
    response = generate_response(inst_model, tokenizer, p)
    print(f"  Q: {p}")
    print(f"  A: {response}\n")


# ===============================================================
# FINAL SUMMARY
# ===============================================================
print("=" * 55)
print("PROJECT COMPLETE: THE FULL LLM PIPELINE")
print("=" * 55)
print("""
Here is everything we built, end to end:

  Stage 1 — Tokenization
    Text → token IDs via Byte Pair Encoding (BPE)

  Stage 2 — Embeddings
    Token IDs → dense vectors (token + position embeddings)

  Stage 3 — Self-Attention
    Each token attends to all previous tokens (causal mask)
    Q @ K^T / sqrt(d) → softmax → attention weights → V

  Stage 4 — Multi-Head Attention
    N heads in parallel, each learning different relationships
    Concat outputs + final linear projection

  Stage 5 — GPT Model
    Embedding → N × TransformerBlock (attn + FF + LayerNorm + residual)
    → Final LayerNorm → LM head → logits over vocab

  Stage 6 — Pretraining
    Next-token prediction on large text
    Cross-entropy loss + AdamW optimizer + backpropagation
    Loss decreasing = model learning language

  Stage 7 — Fine-tuning
    Classification: replace LM head with classifier head
    Instruction: masked loss on response tokens only
    Same architecture, task-specific data, shorter training

This is the architecture behind GPT-2, GPT-3, and the
foundations of GPT-4 and Claude. Scale (more layers, wider
embeddings, more data, longer training) is what separates
our demo from production models — not the architecture.
""")
