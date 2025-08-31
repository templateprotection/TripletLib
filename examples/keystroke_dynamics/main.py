from abc import ABC

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from tripletlib.triplet_utils import EmbeddingModel, TripletDataset
from tripletlib.evaluation_utils import compute_eer_from_scores


# -----------------------
# Hyperparameters
# -----------------------
BATCH_SIZE = 256
EMBEDDING_DIM = 128
LEARNING_RATE = 1e-5
MARGIN = 1.0
EPOCHS = 20
TEST_SIZE = 0.25


# -----------------------
# Model definition
# -----------------------
class LinearEmbeddingModel(EmbeddingModel, ABC):
    def __init__(self, input_size, embedding_dim=64, lr=1e-3, margin=1.0, device="cpu", normalize=True):
        super().__init__(embedding_dim, margin=margin, device=device)

        self.normalize = normalize
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


# ------------
# Load dataset
# ------------
print("Loading dataset...")
df = pd.read_csv("cmu_dataset.csv")

# features and labels
X = df[df.columns[3:]].values.astype(np.float32)
y, _ = pd.factorize(df["subject"])  # convert subjects to integers

# split train and test sets by subject (necessary for properly evaluating triplet loss)
unique_subjects = np.unique(y)
num_train_subjects = int(len(unique_subjects) * (1-TEST_SIZE))

train_subjects = unique_subjects[:num_train_subjects]
test_subjects = unique_subjects[num_train_subjects:]

train_mask = np.isin(y, train_subjects)
test_mask = ~train_mask

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# convert to torch tensors and wrap as (feature, label) tuples
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
train_dataset = list(map(tuple, zip(X_train_tensor, y_train)))
test_dataset = list(map(tuple, zip(X_test_tensor, y_test)))


# --------------------------------------
# Create TripletDatasets and DataLoaders
# --------------------------------------
print("Creating TripletDatasets")
triplet_train = TripletDataset(train_dataset)
triplet_test = TripletDataset(test_dataset)

dataloader_train = DataLoader(triplet_train, batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(triplet_test, batch_size=BATCH_SIZE, shuffle=True)


# ---------------------------
# Instantiate and train model
# ---------------------------
print("Training model...")
device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

model = LinearEmbeddingModel(
    input_size=X_train.shape[1],
    embedding_dim=EMBEDDING_DIM,
    lr=LEARNING_RATE,
    margin=MARGIN,
    device=device
)

history = model.fit(
    dataloader_train,
    dataloader_val=dataloader_test,
    epochs=EPOCHS,
    model_out_path="keystroke_embedding_model.pth"
)


# ----------------
# Plot EER history
# ----------------
eers_train = history['eers_train']
eers_val = history['eers_val']

plt.plot(eers_train)
plt.plot(eers_val)
plt.title('EER vs Epochs')
plt.legend(['Train', 'Validation'])
plt.xlabel('Epochs')
plt.ylabel('EER')
plt.savefig('./images/eer_keystroke.png')
plt.show()


# ----------------------------------------------
# Compute embedding distances and plot histogram
# ----------------------------------------------
y_alike, dists = model.predict_triplet_distances(dataloader_test)
eer, eer_threshold = compute_eer_from_scores(y_alike, dists)
same_class_dists = dists[y_alike == 1]
diff_class_dists = dists[y_alike == 0]

plt.figure(figsize=(10, 6))
plt.hist(same_class_dists, bins=50, color='blue', alpha=0.7, label='Same Class')
plt.hist(diff_class_dists, bins=50, color='red', alpha=0.7, label='Different Class')
plt.axvline(eer_threshold, color='green', linestyle='--', label=f'EER = {eer:.4f}, Threshold: {eer_threshold:.4f}')
plt.xlabel('Distance')
plt.ylabel('Count')
plt.title('Histogram of Embedding Distances')
plt.legend()
plt.grid(True)
plt.savefig('./images/histogram_keystroke.png')
plt.show()
