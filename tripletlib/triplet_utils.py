from abc import ABC, abstractmethod
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .evaluation_utils import compute_eer_from_triplets, compute_auc_from_triplets, compute_dists_from_triplets


class EmbeddingModel(torch.nn.Module, ABC):
    """
    Base class for models trained with triplet loss.
    Subclasses must implement forward() to compute embeddings.
    """

    def __init__(self, embedding_dim, margin=1.0, device="cpu"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        # Triplet loss (standard margin-based)
        self.criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)

    @abstractmethod
    def forward(self, x):
        """
        Compute embeddings for input batch x.
        Must be implemented by subclasses.
        """
        pass

    def fit(self, dataloader_train, dataloader_val=None, epochs=10, model_out_path='model_out.pth') -> dict[
        str, list[float]]:
        """
        Train the model using triplet loss.

        Args:
            dataloader_train: iterable yielding (anchor, positive, negative) batches
            dataloader_val: iterable yielding (anchor, positive, negative) batches
            epochs: number of training epochs
            model_out_path: output path for the model

        Returns:
            history: a dictionary containing historical performance metrics of the model across all epochs
        """

        history = dict()
        history['aucs_train'] = []
        history['aucs_val'] = []
        history['losses_train'] = []
        history['losses_val'] = []
        history['eers_train'] = []
        history['eers_val'] = []

        self.to(self.device)
        self.train()

        for epoch in range(epochs):
            epoch_loss_train = 0.0
            epoch_auc_train = 0.0
            epoch_eer_train = 0.0
            for (anchor, positive, negative) in tqdm(dataloader_train, total=len(dataloader_train),
                                                     desc='Training Loop'):
                anchor, positive, negative = (
                    anchor.to(self.device),
                    positive.to(self.device),
                    negative.to(self.device),
                )

                # Forward pass: compute embeddings
                anchor_emb = self.forward(anchor)
                pos_emb = self.forward(positive)
                neg_emb = self.forward(negative)

                # Compute loss
                loss = self.criterion(anchor_emb, pos_emb, neg_emb)

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss_train += loss.item()
                epoch_auc_train += compute_auc_from_triplets(anchor_emb, pos_emb, neg_emb)
                epoch_eer_train += compute_eer_from_triplets(anchor_emb, pos_emb, neg_emb)[0]

            mean_loss_train = epoch_loss_train / len(dataloader_train)
            mean_auc_train = epoch_auc_train / len(dataloader_train)
            mean_eer_train = epoch_eer_train / len(dataloader_train)
            history['eers_train'].append(mean_eer_train)
            history['aucs_train'].append(mean_auc_train)
            history['losses_train'].append(mean_loss_train)
            print(f"Epoch {epoch + 1}: Train Loss: {mean_loss_train:.4f}")

            if dataloader_val is not None:
                self.eval()
                epoch_loss_val = 0.0
                epoch_auc_val = 0.0
                epoch_eer_val = 0.0
                with torch.no_grad():
                    for (anchor, positive, negative) in tqdm(dataloader_val, total=len(dataloader_val),
                                                             desc='Validation Loop'):
                        anchor, positive, negative = (
                            anchor.to(self.device),
                            positive.to(self.device),
                            negative.to(self.device),
                        )

                        # Forward pass: compute embeddings
                        anchor_emb = self.forward(anchor)
                        pos_emb = self.forward(positive)
                        neg_emb = self.forward(negative)

                        # Compute loss
                        loss = self.criterion(anchor_emb, pos_emb, neg_emb)

                        epoch_loss_val += loss.item()
                        epoch_auc_val += compute_auc_from_triplets(anchor_emb, pos_emb, neg_emb)
                        epoch_eer_val += compute_eer_from_triplets(anchor_emb, pos_emb, neg_emb)[0]

                mean_loss_val = epoch_loss_val / len(dataloader_val)
                mean_auc_val = epoch_auc_val / len(dataloader_val)
                mean_eer_val = epoch_eer_val / len(dataloader_val)
                history['eers_val'].append(mean_eer_val)
                history['aucs_val'].append(mean_auc_val)
                history['losses_val'].append(mean_loss_val)
                print(f"Epoch {epoch + 1}: Validation Loss: {mean_loss_val:.4f}")

            torch.save(self.state_dict(), model_out_path)
        return history

    def predict_triplet_distances(self, dataloader):
        total_alike = []
        total_dists = []
        with torch.no_grad():
            for (anchor, positive, negative) in tqdm(dataloader, total=len(dataloader),
                                                     desc='Evaluation Loop'):
                anchor, positive, negative = (
                    anchor.to(self.device),
                    positive.to(self.device),
                    negative.to(self.device),
                )

                # Forward pass: compute embeddings
                anchor_emb = self.forward(anchor)
                pos_emb = self.forward(positive)
                neg_emb = self.forward(negative)
                y_alike, dists = compute_dists_from_triplets(anchor_emb, pos_emb, neg_emb)
                total_alike.extend(y_alike)
                total_dists.extend(dists)

        return np.array(total_alike), np.array(total_dists)


class TripletDataset(Dataset):
    def __init__(self, base_dataset):
        """
        Wraps a basic dataset to generate triplets.

        Args:
            base_dataset: base dataset to be wrapped
        """
        self.base_dataset = base_dataset

        # Build index lists for each class
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(self.base_dataset):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

        self.labels = [label for _, label in self.base_dataset]

    def __getitem__(self, index):
        anchor_sample, anchor_label = self.base_dataset[index]

        # Positive: random sample with the same label (different index)
        positive_index = index
        while positive_index == index:  # TODO: This can get stuck forever, make sure it's possible.
            positive_index = random.choice(self.class_to_indices[anchor_label])
        positive_sample, _ = self.base_dataset[positive_index]

        # Negative: random sample from a different label
        negative_label = random.choice([label for label in self.class_to_indices.keys() if label != anchor_label])
        negative_index = random.choice(self.class_to_indices[negative_label])
        negative_sample, _ = self.base_dataset[negative_index]

        return anchor_sample, positive_sample, negative_sample

    def __len__(self):
        return len(self.base_dataset)
