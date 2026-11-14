# Support Response Generator 🎧⚡

[![CI Tests](https://github.com/YOUR-USERNAME/support-response-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-USERNAME/support-response-generator/actions)
![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Test Coverage](https://img.shields.io/badge/tests-13%20passed%20(100%25)-brightgreen.svg)

An intelligent, privacy-preserving command-line interface (CLI) tool designed for retail customer support agents. Built from statistical first principles, this tool leverages an **Add-$k$ Smoothed $N$-Gram Language Model** with **Jelinek-Mercer Linear Interpolation** and **Nucleus (Top-$p$) Sampling** to provide real-time next-word autocomplete and response continuation directly on a lightweight laptop without requiring expensive GPU compute.

---

## 🎯 What You Are Building

* **Goal:** A high-speed CLI tool that helps customer support agents rapidly draft accurate, on-brand responses to frequent queries (shipping updates, refunds, returns, billing disputes) by generating next-word suggestions based on historical interactions.
* **Done When:** The tool cleans and anonymizes raw customer dialogues, trains a smoothed $n$-gram model, generates context-aware response continuations, and delivers suggestions via an interactive terminal interface with persistent feedback.

---

## ✅ Completed Milestones

- [x] **1. Prepare an anonymized corpus of customer support interactions.**  
      *Implemented comprehensive PII redaction engine in `app/core.py` targeting customer names, email addresses, order IDs (`#ORD-XXXXX`), carrier tracking numbers (`TRK-XXXXXXXX`), phone numbers, dates, and currency amounts. Cleaned and saved to `data/anonymized_corpus.txt` with domain-specific sub-corpora.*
- [x] **2. Implement the n-gram model with smoothing.**  
      *Implemented `SupportNGramLM` with sentence boundary tokens (`<s>`, `</s>`), Add-$k$ (Laplace) smoothing, Jelinek-Mercer tri/bi/unigram linear interpolation, and cross-entropy perplexity evaluation.*
- [x] **3. Create a text generation function to suggest response continuations.**  
      *Engineered `suggest_next_words()` for next-token probability ranking and `generate_continuation()` with temperature-scaled Top-$p$ (nucleus) sampling to produce fluent multi-word responses without greedy loops.*
- [x] **4. Build a command-line interface for agents to get suggestions.**  
      *Shipped a rich terminal CLI in `app/cli.py` featuring `suggest`, `complete`, `interactive` live drafting, `eval`, `template`, `feedback`, and `anonymize` subcommands.*

---

## 🧠 System Architecture

```text
       Raw Support Conversations (Logs & Chat Transcripts)
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │     Milestone 1: PII Anonymization & Scrub   │
         │  [email], [order_id], [tracking_no], [amount]│
         └──────────────────────┬───────────────────────┘
                                │
                        Cleaned Corpus
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │      Milestone 2: Smoothed N-Gram LM         │
         │   • Sliding Context Window (Bigram/Trigram)  │
         │   • Add-k (Laplace) Smoothing                │
         │   • Jelinek-Mercer Linear Interpolation      │
         └──────────────────────┬───────────────────────┘
                                │
                        Transition Matrix
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │       Milestone 3: Generation Engine         │
         │   • Top-k Next Word Probability Ranking      │
         │   • Autoregressive Nucleus (Top-p) Sampling  │
         │   • Temperature-Controlled Softmax Scaling   │
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │     Milestone 4 & Stretch: Agent Assistant   │
         │   • Interactive Live Terminal Console        │
         │   • Domain Corpora Switcher (Returns/Ship)   │
         │   • Persistent Feedback Loop (+1 / -1)       │
         │   • Dynamic Canned Template Slot Filling     │
         └──────────────────────────────────────────────┘
```

---

## 🚀 Quickstart & Setup

### Requirements
* Python 3.11 or 3.12
* Git

```bash
# 1. Clone repository
git clone https://github.com/YOUR-USERNAME/support-response-generator.git
cd support-response-generator

# 2. Create virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests
pytest
```

---

## 💻 CLI Usage Guide

The tool provides an intuitive CLI via `python run.py <command>` or `python -m app.cli <command>`.

### 1. Verification Check
```bash
python run.py hello
# Output: Support Response Generator is ready. Open README.md for your milestones.
```

### 2. Next-Word Autocomplete (`suggest`)
Get top-$k$ next-token predictions with transition probabilities:
```bash
python run.py suggest "thank you for contacting" --top-k 3
```
*Output:*
```text
Prompt: "thank you for contacting"
Next-word suggestions:
  [1] customer        (prob: 0.5210 |  52.1%) ██████████
  [2] retail          (prob: 0.2840 |  28.4%) █████
  [3] billing         (prob: 0.1420 |  14.2%) ██
```

### 3. Autoregressive Response Continuation (`complete`)
Generate fluent multi-token suggestions with controllable temperature:
```bash
python run.py complete "we have processed your full" --max-tokens 15 --temperature 0.7
```
*Output:*
```text
--- Generated Response ---
we have processed your full refund of [amount] to your original payment method.
```

### 4. Interactive Live Drafting Console (`interactive`)
Launch a real-time assistive drafting interface for support agents:
```bash
python run.py interactive
```

### 5. Model Evaluation & Perplexity Benchmarks (`eval`)
Calculates cross-entropy loss and perplexity on training and held-out validation sets:
```bash
python run.py eval
```
*Output:*
```text
--- Model Evaluation Metrics ---
Vocabulary Size: 208
Bigram Perplexity  - Train: 56.66 | Test: 138.08
Trigram Perplexity - Train: 60.20 | Test: 155.79
```

### 6. PII Scrubber Utility (`anonymize`)
Clean and sanitize arbitrary support logs:
```bash
python run.py anonymize data/raw_support_logs.txt data/anonymized_corpus.txt
```

---

## 🌟 Hackathon Standout Features (Stretch Goals)

### 1. Domain-Specific Corpora Switching (`--category`)
Customer inquiries differ fundamentally across departments. Agents can switch between tailored sub-corpora:
* `shipping`: delivery delays, carrier tracking, dispatch verification.
* `returns`: return approvals, replacement shipments, drop-off locations.
* `billing`: payment receipts, duplicate charges, promotional discounts.
* `general`: greetings, account inquiries, closing remarks.

```bash
python run.py suggest "your delivery has been" --category shipping
python run.py suggest "your return request has" --category returns
```

### 2. Persistent Agent Feedback Loop (`feedback`)
Implements an active learning feedback mechanism using `Store` (`data/app.json`). Agents can log whether a suggestion was helpful:
```bash
# Record feedback
python run.py feedback record --prompt "your order" --suggestion "has" --helpful yes

# View analytics
python run.py feedback stats
```

### 3. Dynamic Macro Templates with Slot Injection (`template`)
Pre-loaded with standardized retail support macros and runtime variable injection:
```bash
# List available templates
python run.py template list

# Render with slot parameters
python run.py template render --name order_shipped --customer "Alex" --order-id "#ORD-48192" --tracking-no "TRK-9810283"
```
*Rendered Output:*
```text
Hello Alex, your order #ORD-48192 has been shipped via express delivery with tracking number TRK-9810283.
```

---

## 📐 Mathematical Formulation

### 1. Maximum Likelihood Estimation (MLE)
For an $n$-gram model, transition probabilities are calculated via empirical frequency ratios:
$$P_{\text{MLE}}(w_i \mid w_{i-n+1}, \dots, w_{i-1}) = \frac{\text{Count}(w_{i-n+1}, \dots, w_i)}{\text{Count}(w_{i-n+1}, \dots, w_{i-1})}$$

### 2. Add-$k$ (Laplace) Smoothing
To prevent zero-probability catastrophes on unseen tokens, probability mass is redistributed across vocabulary $V$:
$$P_{\text{Laplace}}(w \mid \text{ctx}) = \frac{\text{Count}(\text{ctx}, w) + k}{\sum_{w'} \text{Count}(\text{ctx}, w') + k \cdot |V|}$$

### 3. Jelinek-Mercer Linear Interpolation
Balances higher-order context accuracy with lower-order robustness:
$$P_{\text{interp}}(w \mid w_{-2}, w_{-1}) = \lambda_3 P(w \mid w_{-2}, w_{-1}) + \lambda_2 P(w \mid w_{-1}) + \lambda_1 P(w)$$
where $\sum_{i=1}^3 \lambda_i = 1$.

### 4. Perplexity
Measures prediction uncertainty on held-out text sequences of length $N$:
$$\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid \text{context})\right)$$

---

## 🧪 Testing & CI

All milestones and features are covered by automated unit and regression tests in `tests/test_core.py` and `tests/test_smoke.py`.

```bash
pytest -v
```

All 13 tests execute and pass in under 0.25 seconds without external server dependencies.

---

## 📄 License
MIT License. Built for the askThili Language Models Track.
