<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Folder Structure

```
seq2seq-code-generation/
├── README.md
├── requirements.txt
├── notebook.ipynb
├── data/
│   └── README.md
├── checkpoints/
│   └── .gitkeep
├── plots/
│   └── .gitkeep
├── inference/
│   ├── inference.py
│   ├── example_inference.py
│   └── README.md
└── utils/
    ├── __init__.py
    ├── vocabulary.py
    ├── models.py
    └── tokenizer.py
```


***

# README.md

```markdown
# Text-to-Python Code Generation using Seq2Seq Models

A comprehensive implementation and comparison of three sequence-to-sequence models (Vanilla RNN, LSTM, and LSTM with Attention) for generating Python code from natural language descriptions.

## Table of Contents

- [Overview](#overview)
- [Models Implemented](#models-implemented)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Citation](#citation)



## Results

### Quantitative Performance

| Model | BLEU Score | Token Accuracy | Exact Match |
| :-- | :-- | :-- | :-- |
| Vanilla RNN | 10.87 | 18.51% | 0.00% |
| LSTM | 13.60 | 18.68% | 0.00% |
| Attention | 21.85 | 20.29% | 0.00% |

## Overview

This project implements and compares three neural network architectures for the task of generating Python code from natural language docstrings:

1. **Vanilla RNN**: Baseline seq2seq model with simple RNN cells
2. **LSTM**: Improved architecture with Long Short-Term Memory cells for better handling of long-range dependencies
3. **Attention LSTM**: LSTM with attention mechanism to remove the fixed-context bottleneck

The models are trained on the CodeSearchNet Python dataset and evaluated using multiple metrics including BLEU score, token accuracy, and exact match percentage.

## Models Implemented

### 1. Vanilla RNN Seq2Seq
- Simple encoder-decoder architecture
- RNN cells in both encoder and decoder
- Fixed-length context vector bottleneck
- Baseline performance metrics

### 2. LSTM Seq2Seq
- LSTM cells for better long-range dependency handling
- Improved gradient flow during backpropagation
- Better performance on longer sequences

### 3. LSTM with Attention
- Bahdanau-style attention mechanism
- Dynamic context vector for each decoding step
- Significantly improved performance
- Attention weights visualization capability

## Dataset

The project uses the CodeSearchNet Python dataset from Hugging Face:
- Training: 10,000 examples
- Validation: 1,000 examples
- Test: 1,000 examples

Each example consists of:
- **Source**: Natural language docstring (max 50 tokens)
- **Target**: Python code snippet (max 80 tokens)

## Requirements

```

python>=3.8
torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
nltk>=3.6
datasets>=2.0.0
pandas>=1.3.0

```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/seq2seq-code-generation.git
cd seq2seq-code-generation
```


### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```


### 3. Install dependencies

```bash
pip install -r requirements.txt
```


### 4. Download NLTK data (required for BLEU score)

```python
import nltk
nltk.download('punkt')
```


## Training

### Option 1: Using the Jupyter Notebook

1. Open the notebook:
```bash
jupyter notebook notebook.ipynb
```

2. Run all cells sequentially to:
    - Load and preprocess data
    - Build vocabularies
    - Train all three models
    - Evaluate and compare results
    - Generate visualizations

### Option 2: Using Python Scripts (Future Implementation)

```bash
python train.py --model attention --epochs 25 --batch-size 64
```


### Training Configuration

Key hyperparameters used:

- Embedding dimension: 128
- Hidden dimension: 256
- Number of layers: 2
- Dropout: 0.5
- Learning rate: 0.001
- Batch size: 64
- Epochs: 25
- Teacher forcing ratio: 0.5


## Inference

### Basic Inference

```python
import torch
import pickle

# Load vocabularies
with open('vocabularies.pkl', 'rb') as f:
    data = pickle.load(f)
    src_vocab = data['src_vocab']
    trg_vocab = data['trg_vocab']
    config = data['config']

# Load model
from utils.models import AttentionSeq2Seq

model = AttentionSeq2Seq(
    src_vocab_size=config['SRC_V_SIZE'],
    trg_vocab_size=config['TRG_V_SIZE'],
    embed_dim=config['EMBED_DIM'],
    hidden_dim=config['HIDDEN_DIM'],
    num_layers=config['NUM_LAYERS'],
    dropout=0.5,
    pad_idx=config['PAD_IDX']
)

# Load trained weights
checkpoint = torch.load('attention_best.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Tokenize input
from utils.tokenizer import tokenize

docstring = "Calculate the sum of two numbers"
tokens = tokenize(docstring)
src_indices = [config['SOS_IDX']] + src_vocab.numericalize(tokens) + [config['EOS_IDX']]
src_tensor = torch.LongTensor(src_indices).unsqueeze(0)

# Generate code
with torch.no_grad():
    generated_indices = generate_sequence(model, src_tensor, max_len=80)
    generated_code = ' '.join(trg_vocab.decode(generated_indices))

print("Generated Code:")
print(generated_code)
```


### Using the Inference Script

```bash
cd inference
python example_inference.py --docstring "Sort a list of numbers" --model attention
```


### Inference with Attention Visualization

```python
from inference.inference import generate_with_attention

generated_code, attention_weights = generate_with_attention(
    model, 
    docstring, 
    src_vocab, 
    trg_vocab
)

# Visualize attention
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(attention_weights, cmap='viridis')
plt.xlabel('Source Tokens')
plt.ylabel('Generated Tokens')
plt.title('Attention Weights Visualization')
plt.show()
```


### Key Observations

1. **Attention Mechanism Impact**: The attention model achieves ~11% absolute improvement in BLEU score over the vanilla RNN baseline
2. **Performance vs. Docstring Length**: All models show declining accuracy as input length increases, but the attention model maintains relatively better performance
3. **Common Error Patterns**:
    - Repetitive token generation
    - Syntax errors in complex structures
    - Variable naming inconsistencies
    - Missing closing brackets/parentheses
4. **Attention Patterns**: The attention mechanism successfully learns to focus on relevant parts of the input docstring, though it sometimes attends to padding tokens


## Key Findings

1. **Attention is Critical**: The attention mechanism provides substantial improvements over fixed-context models, especially for longer inputs
2. **LSTM vs. RNN**: LSTM cells provide modest improvements over vanilla RNNs, but the architecture change alone is not sufficient for high-quality code generation
3. **Sequence Length Matters**: All models struggle with longer sequences, suggesting the need for:
    - More sophisticated attention mechanisms
    - Larger model capacity
    - More training data
4. **Code-Specific Challenges**:
    - Syntactic correctness is difficult to enforce
    - Variable naming requires semantic understanding
    - Indentation and structure are partially learned

## Limitations

1. **Low Exact Match**: No model achieves exact matches on the test set, indicating that generating perfect code remains challenging
2. **Syntactic Errors**: Generated code often contains syntax errors, particularly with:
    - Bracket matching
    - Indentation
    - Function definitions
3. **Training Data Size**: 10,000 examples may be insufficient for learning complex code patterns
4. **Evaluation Metrics**: BLEU score may not fully capture code quality; execution-based metrics would be more appropriate
5. **Vocabulary Size**: Limited vocabulary (2,302 source, 4,691 target tokens) may restrict the model's expressiveness

## Future Work

1. **Model Architecture Improvements**:
    - Implement Transformer-based models (e.g., CodeBERT, CodeT5)
    - Explore copy mechanisms for handling out-of-vocabulary tokens
    - Add pointer networks for variable name generation
2. **Training Enhancements**:
    - Increase dataset size
    - Implement curriculum learning
    - Use reinforcement learning with execution feedback
3. **Evaluation**:
    - Add execution-based metrics (e.g., pass@k)
    - Implement syntax checking
    - Compare with state-of-the-art code generation models
4. **Deployment**:
    - Create a web interface for interactive code generation
    - Optimize models for inference speed
    - Implement beam search for better generation quality

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{seq2seq-code-generation,
  author = {Your Name},
  title = {Text-to-Python Code Generation using Seq2Seq Models},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/seq2seq-code-generation}
}
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CodeSearchNet dataset: Husain et al., "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search" (2019)
- Attention mechanism: Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
- PyTorch framework and community


## Contact

For questions or collaborations:

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

Last updated: February 13, 2026

```

***

# requirements.txt

```

torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
nltk>=3.6
datasets>=2.0.0
pandas>=1.3.0
tqdm>=4.60.0

```

***

# inference/README.md

```markdown
# Inference Guide

This directory contains utilities for performing inference with trained models.

## Quick Start

### 1. Basic Inference

```python
from inference import generate_code

generated_code = generate_code(
    docstring="Calculate factorial of a number",
    model_path="attention_best.pt",
    vocab_path="vocabularies.pkl"
)
print(generated_code)
```


### 2. Command Line Usage

```bash
python example_inference.py \
    --docstring "Sort a list of integers" \
    --model attention \
    --checkpoint ../checkpoints/attention_best.pt \
    --vocab ../vocabularies.pkl
```


### 3. Batch Inference

```python
from inference import batch_generate

docstrings = [
    "Add two numbers",
    "Sort a list",
    "Find maximum element"
]

results = batch_generate(docstrings, model_path, vocab_path)
for doc, code in zip(docstrings, results):
    print(f"Input: {doc}")
    print(f"Output: {code}\n")
```


## API Reference

### generate_code()

Generate Python code from a natural language description.

**Parameters:**

- `docstring` (str): Natural language description
- `model_path` (str): Path to model checkpoint
- `vocab_path` (str): Path to vocabulary file
- `max_length` (int, optional): Maximum generation length (default: 80)
- `device` (str, optional): 'cpu' or 'cuda' (default: 'cpu')

**Returns:**

- `str`: Generated Python code


### generate_with_attention()

Generate code and return attention weights for visualization.

**Parameters:**

- Same as `generate_code()`
- `return_attention` (bool): If True, returns attention weights

**Returns:**

- `tuple`: (generated_code, attention_weights)


## Examples

See `example_inference.py` for complete working examples.

## Troubleshooting

**Issue**: "Vocabulary file not found"
**Solution**: Make sure you've downloaded `vocabularies.pkl` from the training notebook

**Issue**: "Model architecture mismatch"
**Solution**: Ensure the model checkpoint matches the model type (vanilla_rnn, lstm, or attention)

**Issue**: "Out of memory"
**Solution**: Use CPU inference or reduce batch size

```

***

