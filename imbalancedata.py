# =========================
# 1. Install & Imports
# =========================
# pip install imbalanced-learn scikit-learn pandas

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE


# =========================
# 2. Create Imbalanced Dataset
# =========================
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    weights=[0.9, 0.1],   # Imbalance
    random_state=42
)

print("Original Dataset Shape:", Counter(y))


# =========================
# 3. Train-Test Split (IMPORTANT)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain Shape Before Resampling:", Counter(y_train))
print("Test Shape:", Counter(y_test))


# =========================
# 4. Define Function to Train Model
# =========================
def train_and_evaluate(X_tr, y_tr, X_te, y_te, title):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)

    print(f"\n===== {title} =====")
    print("Accuracy:", accuracy_score(y_te, y_pred))
    print(classification_report(y_te, y_pred))


# =========================
# 5. WITHOUT RESAMPLING (Baseline)
# =========================
train_and_evaluate(X_train, y_train, X_test, y_test, "Baseline (Imbalanced)")


# =========================
# 6. RANDOM UNDERSAMPLING
# =========================
rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train, y_train)

print("\nAfter Undersampling:", Counter(y_under))

train_and_evaluate(X_under, y_under, X_test, y_test, "Random Undersampling")


# =========================
# 7. RANDOM OVERSAMPLING
# =========================
ros = RandomOverSampler(random_state=42)
X_over, y_over = ros.fit_resample(X_train, y_train)

print("\nAfter Oversampling:", Counter(y_over))

train_and_evaluate(X_over, y_over, X_test, y_test, "Random Oversampling")


# =========================
# 8. SMOTE (BEST METHOD)
# =========================
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:", Counter(y_smote))

train_and_evaluate(X_smote, y_smote, X_test, y_test, "SMOTE")


# =========================
# 9. FINAL COMPARISON NOTE
# =========================
print("\n✅ Done! Compare precision, recall, f1-score for best method.")