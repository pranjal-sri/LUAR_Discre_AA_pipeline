# DiscRE + LUAR End-2-End


A Python-based project that combines LUAR (Language Understanding and Representation) and DiscRE (Discourse Relation Embeddings) for author embedding generation.

## Overview

This project provides functionality to:
1. Generate LUAR embeddings from text data
2. Generate DiscRE embeddings from text data
3. Combine both embeddings using a deep embedding network model

## Requirements

The project has two separate requirements files:
- `requirements_pranjal.txt`
- `requirements_vasudha.txt`

Key dependencies include:
- PyTorch
- Transformers
- Pandas
- NumPy

You also need to install git-lfs for it to work.

## Running the code
Download the pretrained models:
```bash
python download_pretrained_models.py
```


Then run the example main file that takes in a message csv file, and returns the embeddings that are saved as a pickle.

```
python main.py
```

## Input Data Format

The input data should be a CSV file with at least two columns:
- `documentID`: Unique identifier for each document
- `message`: Text content to be embedded

## Usage

1. Basic usage with default settings:
```python
python main.py dummy_texts.csv
```

This will:
- Generate LUAR embeddings (512-dimensional)
- Generate DiscRE embeddings (845-dimensional)
- Combine them using a deep embedding model
- Save the results in pickle format

2. The process generates three pickle files:
- `luar_embeddings.pkl`: LUAR embeddings
- `discre_embeddings.pkl`: DiscRE embeddings
- `embeddings.pkl`: Combined embeddings

## Output

- LUAR embeddings: Dictionary mapping document IDs to 512-dimensional vectors
- DiscRE embeddings: Dictionary mapping document IDs to 845-dimensional vectors
- Combined embeddings: Dictionary mapping document IDs to the final embedded representations
