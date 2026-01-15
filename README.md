#  Seq2Seq Code Generation

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Text-to-Python code generation using RNN, LSTM, and Attention mechanisms

**Course Assignment**: Sequence-to-Sequence Learning  
**Author**: Rony Majumder
**Date**: January 2026

---

##  Overview

This project implements three seq2seq architectures for generating Python code from natural language descriptions:

-  **Vanilla RNN Seq2Seq** (Baseline)
-  **LSTM Seq2Seq** (Improved long-term dependencies)
-  **LSTM + Bahdanau Attention** (Dynamic context)

### Dataset
- **CodeSearchNet Python** (10,000 training samples)
- Docstrings → Python functions

### Results Summary

| Model | BLEU Score | Val Loss | Parameters |
|-------|-----------|----------|------------|
| Vanilla RNN | 0.34 | 6.076 | ~10M |
| LSTM | 0.19 | 5.998 | ~12M |
| **LSTM + Attention** | **0.29** | **5.954** | **~15M** |

---

##  Quick Start (3 Methods)

### Method 1: Docker (Recommended for Reproducibility) 

**Prerequisites**: Docker Desktop installed

```bash
# 1. Clone repository
git clone https://github.com/rony31416/seq2seq-code-generation.git
cd seq2seq-code-generation

# 2. Download model files (see instructions below)

# 3. Run with Docker Compose
docker-compose up seq2seq

# Or build and run manually
docker build -t seq2seq-code-gen .
docker run -v $(pwd)/models:/app/models -v $(pwd)/results:/app/results seq2seq-code-gen
