"""
Stage 1: Tokenization
---------------------
Before an LLM can process text, it needs to convert text into numbers.
This is called tokenization. We use Byte Pair Encoding (BPE), the same
algorithm used by GPT-2 and GPT-3.
"""

import tiktoken

# Load the GPT-2 tokenizer
# This gives us the same vocabulary GPT-2 was trained with: 50,257 tokens
tokenizer = tiktoken.get_encoding("gpt2")

print("=" * 50)
print("VOCABULARY SIZE")
print("=" * 50)
print(f"The tokenizer knows {tokenizer.n_vocab:,} unique tokens\n")


# ------------------------------------------------------------------
# 1. ENCODING: text → token IDs
# ------------------------------------------------------------------
print("=" * 50)
print("ENCODING: text → token IDs")
print("=" * 50)

text = "Hello, I am building a large language model from scratch!"
token_ids = tokenizer.encode(text)

print(f"Text:      {text}")
print(f"Token IDs: {token_ids}")
print(f"# Tokens:  {len(token_ids)}\n")


# ------------------------------------------------------------------
# 2. DECODING: token IDs → text
# ------------------------------------------------------------------
print("=" * 50)
print("DECODING: token IDs → text")
print("=" * 50)

decoded = tokenizer.decode(token_ids)
print(f"Decoded back: {decoded}\n")


# ------------------------------------------------------------------
# 3. LOOKING INSIDE: what does each token represent?
# ------------------------------------------------------------------
print("=" * 50)
print("WHAT EACH TOKEN LOOKS LIKE")
print("=" * 50)

for token_id in token_ids:
    token_bytes = tokenizer.decode([token_id])
    print(f"  ID {token_id:>6} → '{token_bytes}'")


# ------------------------------------------------------------------
# 4. SUBWORD TOKENIZATION: how rare/unknown words are handled
# ------------------------------------------------------------------
print()
print("=" * 50)
print("SUBWORD TOKENIZATION")
print("=" * 50)

examples = [
    "unbelievable",       # common enough to be one token
    "supercalifragilistic",  # rare, gets broken into pieces
    "ChatGPT",            # proper noun
    "   ",                # whitespace
    "123 + 456 = 579",    # math
]

for word in examples:
    ids = tokenizer.encode(word)
    pieces = [tokenizer.decode([i]) for i in ids]
    print(f"  '{word}'")
    print(f"    → {pieces}")
    print(f"    → IDs: {ids}")
    print()


# ------------------------------------------------------------------
# 5. SPECIAL TOKENS
# ------------------------------------------------------------------
print("=" * 50)
print("SPECIAL TOKENS")
print("=" * 50)

# GPT models use a special token to mark the end of a document
# so the model knows where one text ends and another begins
eot_token = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
print(f"End-of-text token: {eot_token}")
print(f"  This is token ID 50256 — the last token in the vocabulary")
print(f"  It tells the model: 'this document is finished'\n")


# ------------------------------------------------------------------
# 6. WHY THIS MATTERS FOR THE MODEL
# ------------------------------------------------------------------
print("=" * 50)
print("WHY THIS MATTERS")
print("=" * 50)
print("""
The model never sees text — only token IDs.

A sentence like "The cat sat" becomes something like [464, 3797, 3332].
The model learns patterns between these numbers during training.

When we say a model has a 'context window' of 4096 tokens, that means
it can look at up to 4096 of these IDs at once when predicting the next one.

Next up: Embeddings — we convert these integer IDs into vectors
(lists of floating point numbers) that capture meaning.
""")
