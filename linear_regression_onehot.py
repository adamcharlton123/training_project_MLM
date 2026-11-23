"""Predict RNA efficacy from sequence using linear regression.

Run: python linear_regression_onehot.py
Install once: pip install numpy pandas scipy scikit-learn
Put variants_qbinned.tsv in the same folder as this script.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load ONLY the sequences and their scores (the answers to learn).
data = pd.read_csv(Path(__file__).with_name("variants_qbinned.tsv"), sep="\t",
                   usecols=["sequence", "score"])
letters = np.array([list(s.upper()) for s in data["sequence"]])
scores = data["score"].to_numpy()

# 2. Reserve 20% for a fair test. In the original file, every 100 rows
# belong to one cell, so we keep whole cells out of the training data.
groups = np.arange(len(data)) // 100  # Requires the original row order.
train, test = next(GroupShuffleSplit(test_size=0.2, random_state=42)
                   .split(letters, scores, groups))

# 3. Use exactly four one-hot entries per position, ordered A, C, G, T.
# A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1].
# Keep all four columns even when a base is absent from training.
encoder = OneHotEncoder(
    categories=[["A", "C", "G", "T"] for _ in range(letters.shape[1])],
    handle_unknown="error",  # Reject symbols outside the DNA alphabet.
    sparse_output=True, dtype=np.float32)
X_train = encoder.fit_transform(letters[train])
X_test = encoder.transform(letters[test])

# 4. Learn from training sequences, then predict scores for unseen ones.
model = LinearRegression()
model.fit(X_train, scores[train])
predicted = model.predict(X_test)

# 5. Measure prediction quality: R² close to 1 and MAE close to 0 are best.
print("LINEAR REGRESSION (full one-hot sequence)")
print(f"Test RNAs: {len(test):,}")
print(f"R²:  {r2_score(scores[test], predicted):.8f}")
print(f"MAE: {mean_absolute_error(scores[test], predicted):.8f}")

# Optional biological interpretation: largest base-dependent effects.
# Each position has four coefficients; compare only bases seen in training.
effects = model.coef_
starts = np.cumsum([0] + [len(c) for c in encoder.categories_])
observed = [np.unique(letters[train, pos]) for pos in range(letters.shape[1])]
importance = []
for pos, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
    seen = np.isin(encoder.categories_[pos], observed[pos])
    importance.append(np.ptp(effects[a:b][seen]))
print("Top 5 base changes relative to the most common base:")
for pos in np.argsort(importance)[-5:][::-1]:
    bases = encoder.categories_[pos]
    common = pd.Series(letters[train, pos]).mode().iloc[0]
    alternatives = [b for b in observed[pos] if b != common]
    if not alternatives:
        continue
    alternative = alternatives[0]
    coefficients = dict(zip(bases, effects[starts[pos]:starts[pos + 1]]))
    change = coefficients[alternative] - coefficients[common]
    print(f"  Position {pos + 1}: {common}->{alternative}  {change:+.5f}")
