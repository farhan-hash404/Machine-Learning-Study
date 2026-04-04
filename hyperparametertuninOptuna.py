# =========================
# 1. Install Dependencies
# =========================
# pip install optuna imbalanced-learn scikit-learn

import numpy as np
from collections import Counter

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE

import optuna


# =========================
# 2. Create Imbalanced Dataset
# =========================
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    weights=[0.9, 0.1],
    random_state=42
)

print("Original Dataset:", Counter(y))


# =========================
# 3. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train Distribution:", Counter(y_train))
print("Test Distribution:", Counter(y_test))


# =========================
# 4. Apply SMOTE ONLY on Training Data
# =========================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE:", Counter(y_train_smote))


# =========================
# 5. Define Optuna Objective Function
# =========================
def objective(trial):
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }

    model = RandomForestClassifier(**params, random_state=42)

    # Cross-validation score (F1 is better for imbalance)
    score = cross_val_score(
        model,
        X_train_smote,
        y_train_smote,
        cv=3,
        scoring="f1"
    ).mean()

    return score


# =========================
# 6. Run Optuna Optimization
# =========================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("\nBest Trial:")
print("F1 Score:", study.best_value)
print("Best Params:", study.best_params)


# =========================
# 7. Train Final Model with Best Params
# =========================
best_model = RandomForestClassifier(
    **study.best_params,
    random_state=42
)

best_model.fit(X_train_smote, y_train_smote)


# =========================
# 8. Evaluate on Test Data
# =========================
y_pred = best_model.predict(X_test)

print("\n===== FINAL MODEL PERFORMANCE =====")
print(classification_report(y_test, y_pred))


# =========================
# 9. Done
# =========================
print("\n✅ Hyperparameter tuning completed using Optuna!")