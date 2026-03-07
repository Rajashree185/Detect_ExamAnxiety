"""
train.py - Anxiety Classification Model Training
==================================================
Trains a text classifier for exam anxiety detection using
TF-IDF vectorization + PyTorch Neural Network.

This approach works fully offline without needing to download
any large pre-trained models. For production use with BERT,
see train_bert.py (requires internet to download model).

Architecture:
  Input Text -> TF-IDF Vectorization -> Neural Network -> 3 Classes
  Classes: Low (0), Moderate (1), High (2)
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VAL_FILE = os.path.join(DATA_DIR, "val.csv")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "bert_anxiety_model.pt")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# Hyperparameters
NUM_LABELS = 3
MAX_FEATURES = 5000       # TF-IDF vocabulary size
HIDDEN_SIZE = 256
DROPOUT = 0.3
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
LABEL_NAMES = ["Low", "Moderate", "High"]


# ─── Neural Network Model ───────────────────────────────────────────────────

class AnxietyClassifier(nn.Module):
    """Deep Neural Network for anxiety text classification."""

    def __init__(self, input_size, hidden_size, num_labels, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(hidden_size // 4, num_labels)
        )

    def forward(self, x):
        return self.network(x)


# ─── Dataset Class ───────────────────────────────────────────────────────────

class AnxietyDataset(Dataset):
    """PyTorch Dataset for TF-IDF vectorized anxiety text."""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features.toarray() if hasattr(features, 'toarray') else features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"features": self.features[idx], "label": self.labels[idx]}


# ─── Training Functions ─────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_labels, all_preds


# ─── Main Training Pipeline ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AI Anxiety Classifier — Training Pipeline")
    print("  (TF-IDF + Deep Neural Network)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Load Data ────────────────────────────────────────────────
    print("\n[1/5] Loading preprocessed data...")
    if not os.path.exists(TRAIN_FILE):
        print("  ERROR: Run preprocess.py first!")
        sys.exit(1)

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    print(f"  Train: {len(train_df)} samples | Val: {len(val_df)} samples")

    # ── TF-IDF Vectorization ────────────────────────────────────
    print("\n[2/5] Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 3),       # Unigrams, bigrams, trigrams
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,        # Apply log normalization
        strip_accents="unicode"
    )

    X_train = vectorizer.fit_transform(train_df["text"].values)
    X_val = vectorizer.transform(val_df["text"].values)
    y_train = train_df["label_encoded"].values
    y_val = val_df["label_encoded"].values

    input_size = X_train.shape[1]
    print(f"  Vocabulary size: {input_size}")
    print(f"  Feature matrix: {X_train.shape}")

    # Save vectorizer
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  Vectorizer saved to: {VECTORIZER_PATH}")

    # ── Create DataLoaders ──────────────────────────────────────
    print("\n[3/5] Preparing data loaders...")
    train_dataset = AnxietyDataset(X_train, y_train)
    val_dataset = AnxietyDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Build Model ─────────────────────────────────────────────
    print(f"\n[4/5] Building neural network...")
    model = AnxietyClassifier(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_labels=NUM_LABELS,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {input_size} -> {HIDDEN_SIZE} -> {HIDDEN_SIZE//2} -> {HIDDEN_SIZE//4} -> {NUM_LABELS}")
    print(f"  Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    # ── Training Loop ───────────────────────────────────────────
    print(f"\n[5/5] Training for {NUM_EPOCHS} epochs...")
    print("-" * 60)

    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_labels, val_preds = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # Print every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == NUM_EPOCHS:
            print(f"  Epoch {epoch+1:>3}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size": input_size,
                "hidden_size": HIDDEN_SIZE,
                "num_labels": NUM_LABELS,
                "dropout": DROPOUT,
                "label_names": LABEL_NAMES,
                "val_accuracy": val_acc,
                "epoch": epoch + 1,
                "model_type": "tfidf_nn"
            }, MODEL_SAVE_PATH)

    print("-" * 60)

    # ── Final Report ────────────────────────────────────────────
    # Reload best model for final eval
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, val_labels, val_preds = evaluate(model, val_loader, criterion, device)

    print(f"\n  Final Classification Report (Best Model - Epoch {checkpoint['epoch']}):")
    print(classification_report(val_labels, val_preds, target_names=LABEL_NAMES))

    print("=" * 60)
    print(f"  [OK] Training complete!")
    print(f"  [OK] Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"  [OK] Model saved to: {MODEL_SAVE_PATH}")
    print(f"  [OK] Vectorizer saved to: {VECTORIZER_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
