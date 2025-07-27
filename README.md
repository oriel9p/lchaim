# LCHAIM: Investigating Long Context Reasoning in Hebrew

[![ACL 2025 Findings](https://img.shields.io/badge/ACL_2025-Findings-blue)](https://aclanthology.org/2025.findings-acl.413/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LCHAIM (Long Context Hebrew with Advanced reasoning Inference Model Benchmark)** is the first Hebrew benchmark designed to evaluate Natural Language Inference (NLI) over long contexts requiring complex reasoning skills such as coreference, temporal, logical, and analytical inference.

📍 Official repository for the paper:

> _LCHAIM - Investigating Long Context Reasoning in Hebrew_  
> Ehud Malul*, Oriel Perets*, Ziv Mor, Yigal Kassel, Elior Sulem  
> 📍 Findings of the Association for Computational Linguistics: ACL 2025  
> [📄 Paper Link](https://aclanthology.org/2025.findings-acl.413/)

---

## 🧠 What is LCHAIM?

LCHAIM is a Hebrew translation and validation of the [ConTRoL](https://aclanthology.org/2021.aaai.164/) dataset, tailored to assess the capabilities of Hebrew language models on NLI tasks involving:

- Long premise passages (multi-paragraph)
- Complex reasoning:
  - Coreferential
  - Temporal
  - Logical
  - Analytical

The dataset contains **8,325** Hebrew premise-hypothesis pairs, labeled as:
- **Entailment**
- **Contradiction**
- **Neutral**


---

## 📊 Benchmarked Models

We evaluated:
- 🧠 **AlephBERT**
- 🦸 **LongHero**
- 🤖 **LLMs**: GPT-4o, Dicta-LM 2.0, Gemma-9B

Best performance (52% accuracy) was achieved by **LongHero fine-tuned on HebNLI and LCHAIM**. Human accuracy was ~85%, showing a significant gap in Hebrew NLU.

---
