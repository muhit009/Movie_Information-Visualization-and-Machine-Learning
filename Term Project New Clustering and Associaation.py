#%% Importing Important Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tools import categorical
from prettytable import PrettyTable
import re
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from scipy import stats


#%% Dataset
df = pd.read_csv('Machine Learning Term Project/combined_genres.csv')
print(df.head())
#%% Duplicate Values
print(f' Before Removing Duplicates {df.duplicated().sum()}')
df = df.drop_duplicates()
print(f' After Removing Duplicates {df.duplicated().sum()}')
#%% Null Values with dtypes
null_info = pd.DataFrame({
    'Null Values': df.isnull().sum(),
    'Null Percentage': (df.isnull().sum() / len(df)) * 100,
    'Data Type': df.dtypes
})
print(null_info)
#%% Handling Unwanted and NUll Values
unwanted_values = ['I', 'II', 'V', 'III', 'VII', 'IV', 'XXIII', 'IX', 'XV', 'VI', 'X', 'XIV', 'XIX', 'XXIX', 'XXI', 'VIII', 'XI', 'XVIII', 'XII', 'XIII', 'LXXI', 'XVI', 'XX', 'XXXIII', 'XXXII', 'XXXVI', 'XVII', 'LXIV', 'LXII', 'LXVIII', 'XL', 'XXXIV', 'XXXI', 'XLV', 'XLIV', 'XXIV', 'XXVII', 'LX', 'XXV', 'XXXIX', '2029', 'XXVIII', 'XXX', 'LXXII', '1909', 'XXXVIII', 'XXII', 'LVI', 'LVII' 'XLI', 'LII', 'XXXVII', 'LIX', 'LVIII', 'LXX', 'XLIII', 'XLIX', 'LXXIV', 'XXVI', 'C', 'XLI', 'LVII', 'LV','XLVI', 'LXXVII', 'XXXV', 'LIV', 'LI', 'LXXXII', 'XCIX', 'LXIII']

df = df[~df['year'].astype(str).isin(unwanted_values)]
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)
#%% Clean runtime
def clean_runtime(runtime):
    try:

        return float(re.findall(r'\d+', str(runtime))[0])
    except (IndexError, ValueError):
        return None

df['runtime'] = df['runtime'].apply(clean_runtime)
df = df.dropna(subset=['runtime'])
df['runtime'] = df['runtime'].astype(float)
#%% Clean director and stars
def clean_string(value):
    if pd.isnull(value):
        return value
    return str(value).strip()
df['director'] = df['director'].str.replace('\n', '')
df['star'] = df['star'].str.replace('\n', '')
df['director'] = df['director'].apply(clean_string)
df['star'] = df['star'].apply(clean_string)

print(df[['director', 'star']].head())
#%%
df =df.dropna(subset=['director','star'])
#%%
df = df.dropna(subset=['votes', 'rating'])
print(f"New dataset shape: {df.shape}")


#%% Dropping columns with no importance
df = df.drop(columns=['movie_name','movie_id', 'director_id', 'star_id', 'description'])
print(df.head())
#%% Dropping columns with more than 50% null values
df =df.drop(columns=['certificate','gross(in $)'])
#%%
data_cleaned = df.copy()
#%%
numerical_columns =  data_cleaned.select_dtypes(include =['int64', 'float64'])
#%% Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
data_cleaned['director'] = label_encoder.fit_transform(data_cleaned['director'])
data_cleaned['star'] = label_encoder.fit_transform(data_cleaned['star'])

# One-hot encode 'genre'
one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')  # Avoid dummy variable trap
genre_encoded = one_hot_encoder.fit_transform(data_cleaned[['genre']])

genre_columns = one_hot_encoder.get_feature_names_out(['genre'])
data_cleaned = data_cleaned.drop(columns=['genre'])  # Drop the original 'genre' column
genre_encoded_df = pd.DataFrame(genre_encoded, columns=genre_columns, index=data_cleaned.index)
data_cleaned = pd.concat([data_cleaned, genre_encoded_df], axis=1)

updated_shape = data_cleaned.shape
updated_head = data_cleaned.head()

#%% Resample the data
from sklearn.utils import resample


data_cleaned['genre_combined'] = data_cleaned.filter(like='genre_').idxmax(axis=1)
balanced_data = []
target_size = 50000 // data_cleaned['genre_combined'].nunique()
for genre in data_cleaned['genre_combined'].unique():
    genre_data = data_cleaned[data_cleaned['genre_combined'] == genre]
    resampled_genre_data = resample(genre_data, replace=len(genre_data) < target_size,
                                    n_samples=target_size, random_state=42)
    balanced_data.append(resampled_genre_data)

balanced_data = pd.concat(balanced_data)

balanced_data = balanced_data.drop(columns=['genre_combined'])

balanced_shape = balanced_data.shape
balanced_genre_counts = balanced_data.filter(like='genre_').sum()

#%% Standardization
from sklearn.preprocessing import StandardScaler
numerical_columns = ['year', 'runtime', 'rating', 'votes']
scaler = StandardScaler()
balanced_data[numerical_columns] = scaler.fit_transform(balanced_data[numerical_columns])

standardized_summary = balanced_data[numerical_columns].describe()
print(standardized_summary)

# Dimensionality reduction
#%% Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
X_vif = balanced_data.select_dtypes(include=['float64', 'int64'])  # Include only numeric columns
vif_data = pd.DataFrame()
vif_data['feature'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
while vif_data['VIF'].max() > 10:
    high_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'feature']
    X_vif = X_vif.drop(columns=[high_vif_feature])
    vif_data = pd.DataFrame()
    vif_data['feature'] = X_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print("Final VIF Data:")
print(vif_data)

#%% Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_vif)

print(f"PCA Explained Variance Ratios: {pca.explained_variance_ratio_}")
print(f"Number of Components Retained: {pca.n_components_}")

#%% Singular Value Decomposition (SVD)
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=min(X_vif.shape[1], 6))  # Retain top components
X_svd = svd.fit_transform(X_vif)

print(f"SVD Explained Variance Ratios: {svd.explained_variance_ratio_}")

#%% Summary

dimensionality_reduction_results = {
    "VIF": vif_data,
    "PCA Explained Variance Ratio": pca.explained_variance_ratio_,
    "SVD Explained Variance Ratio": svd.explained_variance_ratio_
}

print("Dimensionality Reduction Summary:")
for key, value in dimensionality_reduction_results.items():
    print(f"{key}:")
    print(value)

#%%
pca = PCA(n_components=2)  # Reduce to 2 components for clustering
X_pca = pca.fit_transform(balanced_data)

print(f"PCA Explained Variance Ratios: {pca.explained_variance_ratio_}")

#%% Step 2: Outlier Detection using IQR
def detect_outliers_iqr(data):
    """Detects outliers using the IQR method."""
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
    return mask

outlier_mask = detect_outliers_iqr(X_pca)
X_pca_cleaned = X_pca[outlier_mask]

print(f"Original data shape: {X_pca.shape}")
print(f"Cleaned data shape: {X_pca_cleaned.shape}")


#%% Step 2: Outlier Detection using Z-Score
from scipy.stats import zscore

z_scores = np.abs(zscore(X_pca))

threshold = 3
outlier_mask = (z_scores < threshold).all(axis=1)

X_pca_cleaned = X_pca[outlier_mask]

print(f"Original data shape: {X_pca.shape}")
print(f"Cleaned data shape: {X_pca_cleaned.shape}")

#%%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# %%  Define range of K values
range_k = range(2, 11)
inertia = []
silhouette_scores = []

for k in range_k:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_pca_cleaned)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca_cleaned, kmeans.labels_))

# %% Plot Elbow Method (Within-Cluster Variation)
plt.figure(figsize=(8, 5))
plt.plot(range_k, inertia, marker='o', linestyle='--')
plt.title('Elbow Method: Within-Cluster Variation vs. K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Variation (Inertia)')
plt.xticks(range_k)
plt.grid()
plt.show()

# %%  Plot Silhouette Analysis
plt.figure(figsize=(8, 5))
plt.plot(range_k, silhouette_scores, marker='o', linestyle='--', color='orange')
plt.title('Silhouette Analysis: Silhouette Score vs. K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(range_k)
plt.grid()
plt.show()

# %%  Optimal K and Final K-Means Clustering
optimal_k = range_k[np.argmax(silhouette_scores)]
print(f"Optimal Number of Clusters (K): {optimal_k}")
final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
final_kmeans.fit(X_pca_cleaned)
cluster_labels = final_kmeans.labels_

# %%  Visualize Final Clusters
plt.figure(figsize=(8, 5))
plt.scatter(X_pca_cleaned[:, 0], X_pca_cleaned[:, 1], c=cluster_labels, cmap='viridis', s=30)
plt.scatter(final_kmeans.cluster_centers_[:, 0], final_kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200,
            label='Centroids')
plt.title(f"Final Clusters with K={optimal_k}")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()

#%% DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


#%%  k-Distance Plot for Optimal Eps

k = 4
nearest_neighbors = NearestNeighbors(n_neighbors=k)
neighbors = nearest_neighbors.fit(X_pca_cleaned)
distances, indices = neighbors.kneighbors(X_pca_cleaned)

distances = np.sort(distances[:, k - 1], axis=0)
plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.title("k-Distance Graph")
plt.xlabel("Data Points (sorted)")
plt.ylabel("k-Distance")
plt.grid()
plt.show()



#%% Apply DBSCAN

eps = 800
min_samples = 5

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(X_pca_cleaned)

dbscan_labels = dbscan.labels_

#%%  Visualize DBSCAN Clusters

plt.figure(figsize=(8, 5))
plt.scatter(X_pca_cleaned[:, 0], X_pca_cleaned[:, 1], c=dbscan_labels, cmap='viridis', s=30)
plt.title("DBSCAN Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()
plt.show()

#%% Summary of Clusters

n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"Number of Clusters: {n_clusters}")
print(f"Number of Noise Points: {n_noise}")




#%% Apriori
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import Binarizer
svd_columns = [f"component_{i}" for i in range(X_svd.shape[1])]
svd_df = pd.DataFrame(X_svd, columns=svd_columns)

#%%
binarizer = Binarizer(threshold=0.0)
binary_data = binarizer.fit_transform(svd_df)
binary_df = pd.DataFrame(binary_data, columns=svd_columns)

print("Binarized Data (Ready for Apriori):")
print(binary_df.head())

#%%
frequent_itemsets = apriori(binary_df, min_support=0.2, use_colnames=True, verbose=1)

print("Frequent Itemsets:")
print(frequent_itemsets)

#%%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

print("Association Rules:")
print(rules.to_string())

#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(rules['support'], rules['confidence'], alpha=0.6, c=rules['lift'], cmap='viridis')
plt.colorbar(label='Lift')
plt.title('Support vs Confidence')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.grid()
plt.show()