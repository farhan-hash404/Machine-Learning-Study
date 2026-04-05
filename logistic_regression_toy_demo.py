import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

def main():
    # Load toy dataset: Iris (multi-class classification)
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print("Dataset loaded: Iris (150 samples, 4 features, 3 classes)")
    print(f"Features: {feature_names}")
    print(f"Classes: {target_names}")
    
    # Feature Engineering
    print("\n--- Feature Engineering ---")
    
    # 1. Polynomial features (degree 2 for interactions/non-linearity)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    poly_features = poly.get_feature_names_out(feature_names)
    print(f"Added polynomial features: {X.shape[1]} -> {X_poly.shape[1]} features")
    
    # 2. Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train/Test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
    
    # Train Logistic Regression (multinomial for multi-class)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Feature Coefficients (top 10 for visualization, since poly expands to many)
    coef_df = pd.DataFrame({
        'feature': poly_features[:20],  # Show first 20
        'coef_setosa': model.coef_[0][:20],
        'coef_versicolor': model.coef_[1][:20],
        'coef_virginica': model.coef_[2][:20]
    })
    print("\nTop Feature Coefficients (first 20 poly features):")
    print(coef_df.round(3))
    
    # Plot coefficients magnitude for setosa vs others (simplified)
    plt.figure(figsize=(10, 6))
    top_idx = np.argsort(np.abs(model.coef_[0]))[-10:]
    plt.barh(poly_features[top_idx], np.abs(model.coef_[0][top_idx]))
    plt.title('Top 10 Feature Coefficients Magnitude (Setosa class)')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.show()
    
    print("\nTraining complete! All plots displayed.")

if __name__ == "__main__":
    main()

