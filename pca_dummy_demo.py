"""
PCA (Principal Component Analysis) Demo on Dummy Data

This script demonstrates PCA dimensionality reduction:
- Generates a synthetic correlated dummy dataset (5 features, 300 samples)
- Standardizes features
- Implements PCA manually using NumPy SVD
- Compares with sklearn.decomposition.PCA
- Prints explained variance ratios and cumulative variance
- Visualizes: scree plot and 2D projection of first two principal components

Requirements: numpy, matplotlib, scikit-learn
Run: python pca_dummy_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate synthetic correlated dummy dataset
# 300 samples, 5 features with intentional correlations
n_samples = 300
n_features = 5

# Create base independent features
x1 = np.random.normal(0, 1, n_samples)
x2 = np.random.normal(0, 1, n_samples)
x3 = np.random.normal(0, 1, n_samples)

# Create correlated features
f1 = x1
f2 = x2
f3 = 0.8 * x1 + 0.2 * np.random.normal(0, 1, n_samples)  # correlated with f1
f4 = 0.7 * x2 + 0.3 * np.random.normal(0, 1, n_samples)  # correlated with f2
f5 = 0.5 * x1 + 0.5 * x2 + 0.3 * x3 + 0.2 * np.random.normal(0, 1, n_samples)  # mixed

X = np.column_stack([f1, f2, f3, f4, f5])

print("=" * 60)
print("PCA Demo on Dummy Correlated Data")
print("=" * 60)
print(f"Dataset shape: {X.shape}")
print(f"Features: Feature_1 to Feature_5 (with intentional correlations)")
print()

# Step 2: Standardize the data (zero mean, unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data standardized (zero mean, unit variance).")
print()

# Step 3: Manual PCA implementation using SVD
class PCAManual:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD: X_centered = U * S * Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Principal components (eigenvectors)
        self.components_ = Vt

        # Explained variance (eigenvalues)
        n_samples = X.shape[0]
        self.explained_variance_ = (S ** 2) / (n_samples - 1)

        # Explained variance ratio
        total_var = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    def transform(self, X):
        X_centered = X - self.mean_
        if self.n_components is not None:
            components = self.components_[:self.n_components]
        else:
            components = self.components_
        return np.dot(X_centered, components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Fit manual PCA
manual_pca = PCAManual(n_components=5)
X_manual_pca = manual_pca.fit_transform(X_scaled)

# Step 4: Compare with sklearn PCA
sklearn_pca = PCA(n_components=5)
X_sklearn_pca = sklearn_pca.fit_transform(X_scaled)

# Step 5: Print results
print("-" * 60)
print("Manual PCA Results:")
print("-" * 60)
print("Explained Variance Ratio:")
for i, ratio in enumerate(manual_pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

print(f"\nCumulative Explained Variance:")
cumulative_manual = np.cumsum(manual_pca.explained_variance_ratio_)
for i, cum_var in enumerate(cumulative_manual):
    print(f"  PC1-PC{i+1}: {cum_var:.4f} ({cum_var*100:.2f}%)")

print("\n" + "-" * 60)
print("Sklearn PCA Results:")
print("-" * 60)
print("Explained Variance Ratio:")
for i, ratio in enumerate(sklearn_pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

print(f"\nCumulative Explained Variance:")
cumulative_sklearn = np.cumsum(sklearn_pca.explained_variance_ratio_)
for i, cum_var in enumerate(cumulative_sklearn):
    print(f"  PC1-PC{i+1}: {cum_var:.4f} ({cum_var*100:.2f}%)")

# Verify manual vs sklearn alignment
print("\n" + "-" * 60)
print("Verification: Manual vs Sklearn Alignment")
print("-" * 60)
# Check if explained variance ratios match (allowing for sign flips in components)
variance_match = np.allclose(
    manual_pca.explained_variance_ratio_,
    sklearn_pca.explained_variance_ratio_,
    atol=1e-6
)
print(f"Explained variance ratios match: {variance_match}")

# Check transformed data correlation (should be very high, may differ in sign)
correlation_pc1 = np.corrcoef(X_manual_pca[:, 0], X_sklearn_pca[:, 0])[0, 1]
correlation_pc2 = np.corrcoef(X_manual_pca[:, 1], X_sklearn_pca[:, 1])[0, 1]
print(f"PC1 correlation (manual vs sklearn): {abs(correlation_pc1):.6f}")
print(f"PC2 correlation (manual vs sklearn): {abs(correlation_pc2):.6f}")

# Step 6: Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Scree plot - Manual PCA
axes[0, 0].bar(range(1, 6), manual_pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
axes[0, 0].plot(range(1, 6), cumulative_manual, 'ro-', label='Cumulative')
axes[0, 0].set_xlabel('Principal Component')
axes[0, 0].set_ylabel('Explained Variance Ratio')
axes[0, 0].set_title('Scree Plot - Manual PCA')
axes[0, 0].set_xticks(range(1, 6))
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Scree plot - Sklearn PCA
axes[0, 1].bar(range(1, 6), sklearn_pca.explained_variance_ratio_, alpha=0.7, color='forestgreen')
axes[0, 1].plot(range(1, 6), cumulative_sklearn, 'ro-', label='Cumulative')
axes[0, 1].set_xlabel('Principal Component')
axes[0, 1].set_ylabel('Explained Variance Ratio')
axes[0, 1].set_title('Scree Plot - Sklearn PCA')
axes[0, 1].set_xticks(range(1, 6))
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: 2D Projection - Manual PCA
axes[1, 0].scatter(X_manual_pca[:, 0], X_manual_pca[:, 1], alpha=0.6, c='steelblue', edgecolors='k', linewidth=0.5)
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')
axes[1, 0].set_title('2D Projection - Manual PCA')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: 2D Projection - Sklearn PCA
axes[1, 1].scatter(X_sklearn_pca[:, 0], X_sklearn_pca[:, 1], alpha=0.6, c='forestgreen', edgecolors='k', linewidth=0.5)
axes[1, 1].set_xlabel('PC1')
axes[1, 1].set_ylabel('PC2')
axes[1, 1].set_title('2D Projection - Sklearn PCA')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('PCA Analysis on Dummy Correlated Data', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('pca_dummy_demo.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Summary & Interpretation")
print("=" * 60)
print(f"• Original dimensions: {n_features}")
print(f"• First 2 PCs capture {cumulative_sklearn[1]*100:.1f}% of total variance")
print(f"• First 3 PCs capture {cumulative_sklearn[2]*100:.1f}% of total variance")
print(f"• Dimensionality reduction from 5D to 2D retains most information")
print("\nPlot saved as 'pca_dummy_demo.png'")
print("\nPCA snippet ready for reuse:")
print("""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(your_data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)
""")

print("\nTask complete! PCA demo trained and visualized on dummy data.")
