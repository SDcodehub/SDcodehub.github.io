---
layout: post
title: Bytes → UTF-8 → BPE — why not just number the alphabet?
date: 2025-09-21
author: Sagar Desai
categories: [LLM, Tokenization]
tags: [Unicode, UTF-8, BPE, Tokenizer]
---

## Question

I implemented the tokenizer from the PDF, but I’m still not clear on bytes and the conversion to bytes. Why can’t we just assign numbers to the alphabet? Why does UTF-8 come into the picture? Start from a high level and then go into details.

## TL;DR

- **Models only understand numbers.** We must turn text into numbers.
- **Unicode** gives every character a universal ID (code point).
- **UTF-8** encodes those code points into bytes (numbers 0–255) using 1–4 bytes per character, so any text becomes a small, universal alphabet of 256 possible byte values.
- **BPE** learns frequent byte (or subword) merges to shorten sequences and capture meaningful chunks.
- Result: No OOV at the byte level, plus efficient tokenization of common words/subwords.

---

## From text to numbers (high level)

Language models are mathematical functions. They don’t “see” letters—only numbers. A naïve scheme like `{ 'a': 1, 'b': 2, ... }` breaks down quickly:

1. What about capitals (`A`, `B`), punctuation, digits?
2. What about other languages (e.g., `你好`, `नमस्ते`, `こんにちは`)?
3. What about emojis (🤔)?

You’d need to invent and maintain a globally consistent mapping for every symbol in every language—error-prone and incompatible across systems. That’s exactly what **Unicode** standardizes.

---

## Level 1: Unicode — the universal character dictionary

Unicode assigns a unique integer (a code point) to every character/symbol:

- `a` → 97
- `A` → 65
- `牛` → 29275
- `🤔` → 129300

Now, `a` is always 97, everywhere. But these integers can be large and vary widely, so we still need a consistent, efficient way to store/transmit them as bytes.

---

## Level 2: UTF-8 — the clever byte encoding

**UTF-8** is a variable-length encoding that turns Unicode code points into one or more bytes (each byte is 0–255):

- Basic ASCII (first 128 code points) use exactly 1 byte. For example, `A` (65) is stored as the single byte `65`.
- Many other characters use 2–4 bytes. For example, `€` is `226, 130, 172` in UTF-8.

Key advantages:

- Any text in any language becomes a sequence of bytes from a fixed, tiny alphabet of 256 values.
- There are no out-of-vocabulary errors at the byte level because every character can be represented as bytes.

---

## Level 3: BPE — from bytes to meaningful chunks

If we tokenize by raw bytes, sequences get long (e.g., `hello` → `[104, 101, 108, 108, 111]`). **Byte-Pair Encoding (BPE)** learns to merge frequent adjacent pairs so common patterns become single tokens.

Training loop (conceptually):

1. Start with the 256 single-byte tokens.
2. Count frequency of adjacent pairs in the corpus.
3. Merge the most frequent pair into a new token (e.g., bytes for `t` and `h` → a token representing `th`).
4. Repeat until you reach the desired vocabulary size.

This compresses common sequences (e.g., " the", "ing", "ation") into single tokens, shortening inputs while preserving a fallback path: rare or novel words decompose into known sub-parts.

---

## How this maps to code (what your tokenizer stores)

- `vocab`: maps token IDs to their byte sequences. It starts with `{0: b'\x00', 1: b'\x01', ..., 255: b'\xff'}` and then adds merged multi-byte tokens like `{256: b'th', ...}`.
- `merges`: the learned, ordered list of pair merges, e.g., `[(b't', b'h'), (b'e', b'r'), ...]`. Order matters.
- Encoding a new word: split into bytes, then apply `merges` in the learned order until no more merges apply.

---

## End-to-end example

Text: `Hello 🤔`

1) Unicode code points (conceptually): `[72, 101, 108, 108, 111, 32, 129300]`

2) UTF-8 bytes: `[72, 101, 108, 108, 111, 32, 240, 159, 164, 142]`

3) After BPE merges (illustrative):

- Common sequences like `He`, `llo`, and leading-space patterns like `" the"` in general English get merged if they were frequent in training.
- Final token IDs might look like: `[id_for_"He", id_for_"llo", id_for_space, id_for_🤔]` depending on your learned vocabulary.

---

## Why not just number the alphabet?

Because language is bigger than the English alphabet. Unicode + UTF-8 gives a universal, compact byte representation for all text. BPE then builds efficient, meaningful tokens on top of those bytes. Together, they provide universality (no OOV at the base) and efficiency (shorter, semantically useful tokens).

I have been implemented the tokeniser and other things as per the .pdf, I am still not fully clear on the bytes and conversion to bytes,

I am confused with these conversions, why can not we assign the numbers to alphbate why is utf-8 is coming in picture, start from high level and then go into very details of theset hings

Let's break it down, starting from your core question and then diving into the details.

***

## The High-Level View: From Words to Math

At the most basic level, a language model is a very complex mathematical function. It can't understand "A" or "B" or "cat". It only understands numbers. So, the absolute first step is to convert our text into a sequence of numbers.

Your question is: **"Why can't we just assign numbers to the alphabet?"** like `{'a': 1, 'b': 2, ...}`.

This is a great starting point, but it quickly runs into major problems:

1.  **What about capital letters?** Okay, so we add `{'A': 27, 'B': 28, ...}`.
2.  **What about punctuation?** We need numbers for `.` `,` `!` `?` `'` `"` and so on.
3.  **What about numbers?** We need `0-9`.
4.  **What about other languages?** How do we handle `你好` or `नमस्ते` or `こんにちは`?
5.  **What about emojis?** 🤔 What number does this get?

Suddenly, our simple dictionary becomes enormous and impossible to manage. We'd need a number for every single character in every language on Earth. This is known as the **large vocabulary problem**.

More importantly, if our model is training and sees a new character it's never seen before (an **out-of-vocabulary** or **OOV** token), it has no number for it and fails.

This is where the layers of abstraction—Unicode, UTF-8, and finally Byte-Pair Encoding (BPE)—come in to solve these problems elegantly.

***

## Level 1: Unicode - The Universal Character Dictionary

To solve the problem of representing every character, the world agreed on a standard called **Unicode**.

Think of Unicode as a giant, official dictionary for characters. It assigns a unique number, called a **code point**, to every character, symbol, and emoji you can think of.

* `a` -> 97
* `A` -> 65
* `牛` -> 29275
* `🤔` -> 129300


This solves the problem of having a universal standard. Now, `a` is always 97, everywhere. But it creates a new problem: some of these numbers are huge! Storing every character as a large number (like 129300) would be very inefficient, especially for common English text where the numbers are small.

***

## Level 2: UTF-8 - The Clever Storage Format

This brings us to **UTF-8**, which is an *encoding*. An encoding is a method for representing those Unicode code points as computer-readable **bytes**.

A **byte** is the fundamental unit of computer memory. It's a number between **0 and 255**. That's it. Everything on your computer—images, music, text—is ultimately stored as a sequence of these 0-255 numbers.

Here's the genius of UTF-8:

* **It's a variable-length encoding.** It uses a different number of bytes to store different characters.
* For any character in the basic English alphabet and common symbols (the first 128 Unicode code points), UTF-8 uses **just one byte**. The Unicode code point is the same as the byte value. For example, the character 'A' (Unicode point 65) is stored as the single byte `65`.
* For more complex characters from other languages or emojis, it uses a sequence of **two, three, or four bytes**. For example, the '€' symbol is represented by three bytes: `226`, `130`, `172`.

This is the key insight: **Using UTF-8, we can represent *any text in any language* as a sequence of numbers between 0 and 255.**

This is a massive breakthrough for our tokenizer. We now have:
* A **fixed, small base alphabet** of just 256 "characters" (the byte values 0-255).
* **ZERO chance of an "out-of-vocabulary" error** at this level. [cite_start]Any text can be broken down into this universal set of bytes. [cite: 108]

***

## Level 3: BPE - From Bytes to Meaningful Chunks

We've solved the representation problem, but now we have a new one: our text sequences are very long. The word `hello` becomes five separate tokens: `[104, 101, 108, 108, 111]`. A long sentence would be hundreds of byte tokens. [cite_start]This makes it hard for the model to see patterns and learn relationships between words. [cite: 123]

This is where the **Byte-Pair Encoding (BPE)** algorithm in your `tokonizer.py` comes in. BPE is a data compression algorithm that cleverly creates a vocabulary of subwords.

Here’s how it works during training:

1.  **Start with the Basics:** The initial vocabulary is just the 256 possible bytes.
2.  **Count Pairs:** The algorithm scans the entire training dataset (e.g., all of TinyStories) and counts how often every adjacent pair of bytes occurs.
3.  **Find the Most Common Pair:** It finds the pair that appears most frequently. For example, in English text, the bytes for `'t'` and `'h'` might appear next to each other very often. Let's say their byte values are `116` and `104`.
4.  **Merge and Create a New Token:** The algorithm "merges" this most frequent pair `(116, 104)` into a single new token. It adds this new token to the vocabulary with the next available ID, say `256`. So, token `256` now represents the subword `th`.
5.  **Repeat:** It goes back to the text, replaces all `[116, 104]` sequences with `[256]`, and then repeats the process. It finds the *next* most common pair (which might now involve the new token `256`, like the pair representing `' the'`) and merges that.

[cite_start]This process is repeated for a set number of merges (e.g., until the vocabulary reaches 10,000 tokens). [cite: 134, 165, 166, 167]

### How This Reflects in Your Code

* `self.vocab`: This is the final dictionary mapping token IDs to the bytes they represent. It starts with `{0: b'\x00', 1: b'\x01', ... 255: b'\xff'}` and then adds the new merged tokens like `{256: b'th', ...}`.
* `self.merges`: This is the ordered list of merge rules that were learned, like `[(b't', b'h'), (b'e', b'r'), ...]`. The order is crucial.
* [cite_start]`_apply_bpe_words` function: When you encode new text, this function takes a word (as bytes), breaks it into individual bytes, and then repeatedly applies the learned `merges` in the exact order they were learned until no more merges can be made. [cite: 274, 275]

### The Final Result

By using BPE on top of UTF-8 bytes, you get the best of both worlds:

* **No OOV errors:** The model can handle any text because its base is the 256 bytes.
* [cite_start]**Efficiency:** Common words and subwords (`" the"`, `"ing"`, `"ation"`) become single tokens, making the input sequences much shorter and more manageable for the model to learn from. [cite: 137]
* **Flexibility:** It can handle new or rare words by breaking them down into known subwords (e.g., "tokenization" might become `["token", "ization"]`).

So, the progression is:
**Text** (`"Hello 🤔"`) -> **Unicode** (`[72, 101, ..., 129300]`) -> **UTF-8 Bytes** (`[72, 101, ..., 240, 159, 164, 142]`) -> **BPE Tokens** (`[id_for_Hello, id_for_🤔]`)