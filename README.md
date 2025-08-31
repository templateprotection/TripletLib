# TripletLib
[![GitHub](https://img.shields.io/badge/GitHub-TripletLib-blue)](https://github.com/templateprotection/TripletLib)

TripletLib is a lightweight Python library for training and evaluating embedding models using **triplet loss**. It is designed to help you build models that learn meaningful embeddings for images, time series, or any vector data, enabling tasks like verification, retrieval, or classification based on similarity.

## What is Triplet Loss?

A "triplet" is a set of 3 samples of data `(anchor, positive, negative)`. The `anchor` is a data sample from a specific class, the `positive` is another sample from the *same* class, and the `negative` is a sample from an entirely different class.
Triplet loss is a type of metric learning objective that trains an embedding model to map inputs into a vector space such that:

- Samples from the **same class** (e.g., anchor and positive) are close together.
- Samples from **different classes** (e.g., anchor and negative) are far apart.

This is particularly useful for tasks like biometric identification (e.g., face recognition), where traditional classification cannot generalize to unseen classes (new faces).

## Features

- Base `EmbeddingModel` class with a built-in training loop.
- `TripletDataset` wrapper to automatically generate triplets from any dataset.
- Utilities for evaluating embeddings with **EER** and **AUC** metrics.
- Compatible with PyTorch and supports GPU acceleration.

## Installation

You can install TripletLib via pip:

```bash
pip install tripletlib
```
## Usage
```python
import torch
from tripletlib.triplet_utils import EmbeddingModel, TripletDataset
from torch.utils.data import DataLoader

# Example: using a simple linear embedding model
class LinearEmbeddingModel(EmbeddingModel):
    def __init__(self, input_size, embedding_dim=64, lr=1e-3, margin=1.0, device="cpu"):
        super().__init__(embedding_dim, margin=margin, device=device)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, embedding_dim)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.fc(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)

# Prepare your dataset as a list of (feature, label) tuples
train_dataset = [(torch.randn(10), label) for label in range(5) for _ in range(20)]
triplet_train = TripletDataset(train_dataset)
dataloader_train = DataLoader(triplet_train, batch_size=16, shuffle=True)

# Instantiate and train the model
model = LinearEmbeddingModel(input_size=10, embedding_dim=64, lr=1e-3)
history = model.fit(dataloader_train, epochs=5)
```

## Examples
You can find examples of real usages in `/examples`, including:
- Keystroke Dynamics: Comparing typing patterns to determine if they are from the same individual.
- More to come...

