---
layout: post
title: TinyStories — Why 10K vs 32K Vocab Gives ~Same Bytes/Token
date: 2025-09-21
author: Sagar Desai
categories: [LLM, Tokenization]
tags: [BPE, Tokenizer, TinyStories]
---

## TL;DR

- **Observation:** On TinyStories, 10K and 32K tokenizers yield almost identical compression (bytes/token).
- **Why:** TinyStories is simple and repetitive; a 10K vocab already saturates frequent patterns, so extra 22K tokens are rarely used.
- **When gap grows:** On complex, diverse corpora (e.g., OpenWebText, code, multilingual), larger vocabs reduce token count and improve bytes/token noticeably.

---

## Reference results

- **TinyStories 10K:** 4.058 bytes/token (10 sampled docs, seed 42)
- **TinyStories 32K:** 4.072 bytes/token (10 sampled docs, seed 42)


## Why the small gap on TinyStories?

1. **Vocabulary saturation:** TinyStories uses a limited, repetitive lexicon. A 10K vocab already contains almost all common words/subwords ("the", "and", "play", simple names), so merges capture most compression.
2. **Diminishing returns:** The extra 22K tokens in a 32K vocab skew toward rare/complex words that barely occur in TinyStories. They aren’t exercised, so compression barely changes.
3. **Short, simple morphology:** Few long rare words means fewer opportunities where a larger vocab would replace multi-token fragments with single tokens.

Analogy: Two toolboxes (100 vs 300 tools) both contain a hammer and screwdriver. For a simple chair, the extra 200 specialty tools don’t make you faster.


## When you will see a bigger gap

- **OpenWebText / web-scale prose:** Broader vocabulary and topic diversity. Larger vocabs have single tokens for many complex words (e.g., transformer, backpropagation, jurisdiction), reducing token count.
- **Code corpora:** Identifiers and symbols benefit from longer subword units; bigger vocabs capture common stems/snippets.
- **Multilingual or domain jargon:** Medical, legal, or multilingual text contains many low-frequency segments that 10K vocabs would split into many pieces.

Result: With complex corpora, a 32K tokenizer often yields fewer tokens for the same bytes, improving bytes/token.


## Notes on the metric

- **Bytes/token** here is the average UTF-8 byte length of the raw text divided by the number of tokens produced by the tokenizer.
- For simple English text, average bytes per character ≈ 1 (ASCII), so differences primarily come from token count, not byte size.
- On multilingual text, characters may be 2–4 bytes each in UTF-8; both numerator and denominator shift.


## Quick repro sketch

```python
import random

def bytes_per_token(texts, encode):
    total_bytes = sum(len(t.encode('utf-8')) for t in texts)
    total_tokens = sum(len(encode(t)) for t in texts)
    return total_bytes / total_tokens

random.seed(42)
sampled = random.sample(tinystories_docs, k=10)

ten_k_bpt = bytes_per_token(sampled, encode_10k)
thirty_two_k_bpt = bytes_per_token(sampled, encode_32k)
print(ten_k_bpt, thirty_two_k_bpt)
```


## Code reference

- Repository: [SDcodehub/assignment1-basics](https://github.com/SDcodehub/assignment1-basics)
- Script: [cs336_basics/compute_bytes_per_token.py](https://github.com/SDcodehub/assignment1-basics/blob/main/cs336_basics/compute_bytes_per_token.py)


## Takeaways

- On TinyStories, 10K is already "good enough"; moving to 32K yields minimal gains.
- Expect a larger gap on complex, real-world text (OpenWebText, code, multilingual).
- If your deployment domain resembles TinyStories (simple, repetitive), a smaller vocab can be sufficient and cheaper. If it resembles the web, prefer the larger vocab.


