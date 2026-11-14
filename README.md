# Support Response Generator

The Support Response Generator is a command-line interface (CLI) tool designed for customer support agents in retail. It leverages next-word prediction to quickly draft responses to common queries, improving efficiency and consistency in support interactions. By analyzing a corpus of past support conversations, the tool provides intelligent suggestions to help agents compose replies faster.

## What you are building

**Goal.** You will build a command-line interface tool that helps customer support agents draft responses by generating next-word suggestions based on past support interactions.

**Done when.** The tool successfully generates next-word suggestions for support agents, using an n-gram model trained on an anonymized corpus of customer support interactions, and provides these suggestions through a command-line interface.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare an anonymized corpus of customer support interactions.**  
      Hint: Use text processing techniques to clean and anonymize the corpus.
- [ ] **2. Implement the n-gram model with smoothing.**  
      Hint: Implement the n-gram model with smoothing, as covered in the 'Build a Next-Word Predictor from Scratch' drill.
- [ ] **3. Create a text generation function to suggest response continuations.**  
      Hint: Develop a function that takes a partial sentence and uses your n-gram model to predict the most probable next words.
- [ ] **4. Build a command-line interface for agents to get suggestions.**  
      Hint: Build a command-line interface using appropriate libraries to allow user input and display suggestions.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow agents to specify different corpora for different types of queries.
- Implement a simple feedback mechanism where agents can mark suggestions as helpful or unhelpful.
- Add a feature to save and load agent-specific response templates alongside the generated suggestions.
