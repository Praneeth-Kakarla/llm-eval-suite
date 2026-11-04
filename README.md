# LLM Evaluation & Safety Test Suite

![LLM Eval](https://github.com/Praneeth-Kakarla/llm-eval-suite/actions/workflows/eval.yml/badge.svg)

An automated test suite for a RAG-style customer-support bot (BreezeFans), built with Python, DeepEval, pytest, and the Google Gemini API. It runs in GitHub Actions on every push.

## What it tests

**Quality (15 golden cases)**
- Answer Relevancy: does the answer address the question?
- Faithfulness: is the answer grounded in the retrieved context (no made-up facts)?
- Scored by an LLM judge (custom Gemini judge for DeepEval).

**Safety (10 cases)**
- Prompt injection: 5 cases checking the system prompt is never leaked
- Instruction override: 1 case
- PII echo: 3 cases (card, Aadhaar, phone number must not be repeated back)
- Out-of-scope: 1 case (the bot must admit it doesn't know)

## Results

| Suite | Result |
|---|---|
| Quality (15 cases) | 12/15 before the retrieval fix, 15/15 after |
| Safety (10 cases) | 10/10 |

## Findings

- **Retrieval bug found by the suite:** keyword matching didn't strip punctuation or handle plurals, so questions like "How can I contact support?" retrieved the wrong context and the bot gave non-answers (3 of 15 cases failed). Fixing tokenization (punctuation, stopwords, plurals) resolved all three.
- **Robustness:** a safety test once hit an empty LLM response (it did not reproduce on rerun). Added retry handling for empty and overloaded (503/429) responses.

- **Test design fix:** the question "Is installation free?" failed relevancy in CI, because the grounded answer also mentions timing and the question did not ask for it. Rewrote the question to ask for both cost and timing instead of lowering the threshold.

## Project structure

| File | Purpose |
|---|---|
| `my_bot.py` | Support bot with a small knowledge base, retrieval, and Gemini calls |
| `judge.py` | Custom Gemini judge for DeepEval |
| `test_quality.py` | 15 golden Q&A cases (relevancy + faithfulness) |
| `test_safety.py` | 10 prompt-injection, PII, and out-of-scope tests |
| `.github/workflows/eval.yml` | CI workflow |

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your_key"
export BOT_MODEL="gemini-3.1-flash-lite"

pytest test_safety.py -v -p no:warnings
pytest test_quality.py -v -p no:warnings
```

The quality suite makes many judge calls and can take around 15 minutes on the free tier.

## CI

The workflow in `.github/workflows/eval.yml` runs both suites on every push and pull request. Add your key as a repository secret named `GOOGLE_API_KEY`.

## Limitations

- Small knowledge base and golden dataset (a demo, not a production benchmark).
- LLM-judge scores are non-deterministic and can vary between runs.
- Substring-based safety checks catch obvious leaks, not every possible one.
