"""
Customer Segmentation using KMeans Clustering

This script demonstrates customer segmentation:
- Generates synthetic customer data (400 samples, 4 clusters) with features:
  - Age (18-70)
  - Annual Income (20k-120k)
  - Spending Score (0-100)
- Uses elbow method to determine optimal k=4
- Fits KMeans, visualizes segments, interprets customer profiles
- Standalone; requires: scikit-learn, numpy, pandas, matplotlib

Run: python customer_segmentation_clustering.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import seaborn as sns  # For better pairplot visualization

# Step 1: Generate synthetic customer data with 4 clusters
# Centers manually set for realistic customer profiles
n_samples = 400
cluster_centers = [
    [25, 30000, 10],   # Young low-income low-spenders
    [45, 60000, 50],   # Middle-aged average income average spenders
    [35, 90000, 90],   # Young high-income high-spenders
    [55, 120000, 20]   # Older high-income low-spenders (conservative)
]
X, true_labels = make_blobs(
    n_samples=n_samples,
    centers=cluster_centers,
    cluster_std=[4, 6, 5, 7],
    n_features=3,
    random_state=42
)

# Scale to realistic ranges
X[:, 0] = np.clip(18 + (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min()) * 52, 18, 70)  # Age
X[:, 1] = np.clip(20000 + (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min()) * 100000, 20000, 120000)  # Income
X[:, 2] = np.clip((X[:, 2] - X[:, 2].min()) / (X[:, 2].max() - X[:, 2].min()) * 100, 0, 100)  # Spending Score

# Create DataFrame
columns = ['Age', 'Annual_Income', 'Spending_Score']
customer_df = pd.DataFrame(X, columns=columns)

print("Dataset shape:", customer_df.shape)
print("\nDataset preview:")
print(customer_df.head())
print("\nDataset statistics:")
print(customer_df.describe())

# Step 2: Elbow Method to find optimal k
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.savefig('elbow_customer_segmentation.png', dpi=150, bbox_inches='tight')
plt.show()

# Step 3: Fit KMeans with k=4
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

customer_df['Cluster'] = cluster_labels

# Step 4: Results
print("\nKMeans Clustering Results:")
print(f"Centroids (Age, Income, Spending Score):")
for i, centroid in enumerate(kmeans.cluster_centers_):
    print(f"  Cluster {i}: Age={centroid[0]:.1f}, Income=${centroid[1]:.0f}, Spending={centroid[2]:.1f}")
print(f"Inertia (WCSS): {kmeans.inertia_:.2f}")

# Interpret segments
print("\nCustomer Segments Interpretation:")
segment_profiles = {
    0: "Young Low-Income Low-Spenders (Budget-conscious students/early career)",
    1: "Middle-Aged Average Income Average Spenders (Family-oriented)",
    2: "Young High-Income High-Spenders (Trendy professionals)",
    3: "Older High-Income Conservative Spenders (Investors/savers)"
}
for label in range(optimal_k):
    count = sum(cluster_labels == label)
    print(f"  {segment_profiles[label]}: {count} customers ({count/n_samples*100:.1f}%)")

# Step 5: Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
scatter_cols = [('Age', 'Annual_Income'), ('Age', 'Spending_Score'), ('Annual_Income', 'Spending_Score'), ('Age', 'Annual_Income')]

for i, (col1, col2) in enumerate(scatter_cols):
    row, col = i // 2, i % 2
    sns.scatterplot(data=customer_df, x=col1, y=col2, hue='Cluster', palette='viridis', ax=axes[row, col])
    axes[row, col].set_title(f'{col1} vs {col2} by Cluster')

plt.suptitle('Customer Segments Visualization', fontsize=16)
plt.tight_layout()
plt.savefig('customer_segments_viz.png', dpi=150, bbox_inches='tight')
plt.show()

# Pairplot for all features
sns.pairplot(customer_df, hue='Cluster', palette='viridis')
plt.suptitle('Customer Segmentation Pairplot', y=1.02)
plt.savefig('customer_segments_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlots saved: 'elbow_customer_segmentation.png', 'customer_segments_viz.png', 'customer_segments_pairplot.png'")
print("\nTask complete! Customer segmentation model trained on dummy data.")

