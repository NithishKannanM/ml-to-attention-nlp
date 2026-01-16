import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pickle

from src.models.cnn_lstm import CNNLSTM
from src.utils.config import *
from src.data.dl_dataset import TextDataset
from src.features.dl_tokenizer import SimpleTokenizer
from src.utils.padding import pad_sequences
from src.data.dataset import load_and_split
from src.evaluate.dl_metrics import evaluate_dl, compute_val_accuracy
from src.utils.plot_training import plot_training


# -------------------------
# Load data
# -------------------------
X_train, X_val, y_train, y_val = load_and_split(
    "dataset/IMDB Dataset.csv",
    preprocess=True
)

# -------------------------
# Tokenizer
# -------------------------
tokenizer = SimpleTokenizer(max_vocab=VOCAB_SIZE)
tokenizer.fit(X_train)

X_train_seq = [tokenizer.encode(t) for t in X_train]
X_val_seq   = [tokenizer.encode(t) for t in X_val]

X_train_seq = pad_sequences(X_train_seq, MAX_LEN)
X_val_seq   = pad_sequences(X_val_seq, MAX_LEN)

y_train = torch.LongTensor(y_train.values)
y_val   = torch.LongTensor(y_val.values)

# -------------------------
# DataLoaders
# -------------------------
train_loader = DataLoader(
    TextDataset(X_train_seq, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TextDataset(X_val_seq, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -------------------------
# Model
# -------------------------
model = CNNLSTM(VOCAB_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

train_losses = []
val_accuracies = []

# -------------------------
# Training loop
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x).squeeze()
        loss = criterion(pred, y.float().view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # ---- epoch metrics ----
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    val_acc = compute_val_accuracy(model, val_loader, DEVICE)
    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

# -------------------------
# Save artifacts
# -------------------------
os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/cnn_lstm.pth")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ CNN-LSTM model and tokenizer saved")

# -------------------------
# Final evaluation + plots
# -------------------------
evaluate_dl(model, val_loader, DEVICE)

plot_training(
    train_losses,
    val_accuracies,
    title="CNN-LSTM Training"
)
