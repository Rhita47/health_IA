import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv("health_dataset.csv")

# Display basic information
print("Dataset Info:")
df.info()
print("\nSummary Statistics:")
print(df.describe())

# Handling missing values (if any)
print("\nMissing Values:")
print(df.isnull().sum())
df = df.dropna()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Standardizing the dataset
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.iloc[:, 1:])  # Exclude first column if it's an identifier

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_scaled)
df['Cluster'] = clusters

# Display cluster means
print("\nCluster Means:")
print(df.groupby('Cluster').mean())

# PCA for Dimensionality Reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

# Scatter plot of PCA results
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Health Data')
plt.show()

# Explained variance
explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance Ratio:", explained_variance)