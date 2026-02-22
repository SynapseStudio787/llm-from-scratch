# LLM From Scratch — Session Context

## What this project is
Ramon is building an LLM from scratch as a learning exercise, following the concepts from *Build a Large Language Model (From Scratch)* by Sebastian Raschka. The goal is to watch a full implementation first, then go through the book independently.

## Teaching style
- Explanation first, then code
- Run code and show output after each stage
- Commit and push to GitHub after each stage
- Ramon is comfortable with Python, new to PyTorch
- Explain PyTorch concepts (tensors, gradients, etc.) as they come up

## GitHub
- Repo: https://github.com/SynapseStudio787/llm-from-scratch.git
- Remote is already configured with auth token in the local git config

## Stages
- [x] Stage 1: Tokenization — `01_tokenization.py`
- [x] Stage 2: Embeddings — `02_embeddings.py`
- [x] Stage 3: Self-Attention with causal masking — `03_attention.py`
- [x] Stage 4: Multi-Head Attention (efficient reshape trick) — `04_multihead_attention.py`
- [x] Stage 5: Full GPT model (TransformerBlock, FeedForward, weight tying, generation) — `05_gpt_model.py`
- [x] Stage 6: Pretraining (cross-entropy loss, AdamW, training loop, before/after generation) — `06_pretraining.py`
- [x] Stage 7: Fine-tuning (classification head + instruction tuning with loss masking) — `07_finetuning.py`

## Resume instructions
Pick up at the next unchecked stage. Follow the same pattern:
1. Explain the concept
2. Write the file (`05_gpt_model.py`, etc.)
3. Run it and walk through the output
4. Commit and push
5. Mark the stage complete in this file
