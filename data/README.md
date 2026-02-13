# data/README.md

```markdown
# Dataset Information

This project uses the CodeSearchNet Python dataset, which is automatically downloaded from Hugging Face when you run the notebook.

## Dataset Statistics

- Total available: 455,243 Python functions
- Used in this project:
  - Training: 10,000 examples
  - Validation: 1,000 examples
  - Test: 1,000 examples

## Manual Download (Optional)

If you want to download the dataset manually:

```python
from datasets import load_dataset

dataset = load_dataset("Nan-Do/code-search-net-python", split="train")
```


## Data Format

Each example contains:

- `func_documentation_string`: Natural language description
- `func_code_string`: Python code implementation


## Preprocessing

The notebook performs the following preprocessing:

1. Tokenization of both docstrings and code
2. Filtering by length (docstring < 150 tokens, code < 200 tokens)
3. Vocabulary building with frequency threshold (min 10 occurrences)
4. Special token handling (PAD, SOS, EOS, UNK)
```
