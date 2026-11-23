"""Predict RNA efficacy from sequence using a small neural network.

Run: python neural_network_onehot.py
Install once: pip install numpy pandas scikit-learn torch
Put variants_qbinned.tsv in the same folder as this script.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load ONLY sequence (input) and score (answer).
data = pd.read_csv(Path(__file__).with_name("variants_qbinned.tsv"), sep="\t",
                   usecols=["sequence", "score"])
letters = np.array([list(s.upper()) for s in data["sequence"]])
scores = data["score"].to_numpy(dtype=np.float32)

# 2. Use EXACTLY the same held-out cells as the linear regression model.
groups = np.arange(len(data)) // 100  # Requires the original row order.
train, test = next(GroupShuffleSplit(test_size=0.2, random_state=42)
                   .split(letters, scores, groups))

# 3. Use exactly four one-hot entries per position, ordered A, C, G, T.
# A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1].
# Keep all four columns even when a base is absent from training.
encoder = OneHotEncoder(
    categories=[["A", "C", "G", "T"] for _ in range(letters.shape[1])],
    handle_unknown="error",  # Reject symbols outside the DNA alphabet.
    sparse_output=False, dtype=np.float32)
X_train = torch.from_numpy(encoder.fit_transform(letters[train]))
X_test = torch.from_numpy(encoder.transform(letters[test]))

# 4. Scale the scores to help the network learn (undo after prediction).
mean, std = scores[train].mean(), scores[train].std()
y_train = torch.from_numpy((scores[train] - mean) / std).unsqueeze(1)

# 5. A tiny neural network: sequence -> 16 neurons -> score.
torch.manual_seed(42)  # Reproducible starting weights and batches.
torch.set_num_threads(4)
model = nn.Sequential(nn.Linear(X_train.shape[1], 16), nn.ReLU(),
                      nn.Linear(16, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()  # Penalises differences from the true score.

# 6. Train for 12 epochs; each epoch sees all training sequences once.
for epoch in range(12):
    for batch in torch.randperm(len(train)).split(1024):
        predicted = model(X_train[batch])
        loss = loss_fn(predicted, y_train[batch])
        optimizer.zero_grad()
        loss.backward()   # Find how each weight should change.
        optimizer.step()   # Update the weights to reduce error.

# 7. Predict held-out scores and measure accuracy on the ORIGINAL scale.
model.eval()
with torch.no_grad():
    predicted = model(X_test).squeeze(1).numpy() * std + mean
print("NEURAL NETWORK (full one-hot sequence; 16 hidden neurons)")
print(f"Test RNAs: {len(test):,}")
print(f"R²:  {r2_score(scores[test], predicted):.8f}")
print(f"MAE: {mean_absolute_error(scores[test], predicted):.8f}")
