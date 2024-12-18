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
df = df.drop(columns=['movie_id', 'director_id', 'star_id', 'description'])
print(df.head())
#%% Dropping columns with more than 50% null values
df =df.drop(columns=['certificate','gross(in $)'])

#%% Aggregation

#%% Before Upsampling and Downsampling
genre_counts = df['genre'].value_counts()

plt.figure(figsize=(16, 10))
genre_counts.plot(kind='bar')
plt.title('Genre Distribution', fontsize=16)
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Number of Movies', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#%% Upsample and Downsample
film_noir_data = df[df['genre'] == 'Film-Noir']
film_noir_upsampled = film_noir_data.sample(n=5000, random_state=42, replace=True)

# Downsample the top 15 genres to 5000 samples
top_15_genres = df['genre'].value_counts().index[:15]
downsampled_data = pd.concat([
    x.sample(n=5000, random_state=42) if len(x) >= 5000 else x
    for _, x in df[df['genre'].isin(top_15_genres)].groupby('genre')
]).reset_index(drop=True)
balanced_data = pd.concat([downsampled_data, film_noir_upsampled], ignore_index=True)
print(f"Balanced dataset shape: {balanced_data.shape}")
print(balanced_data['genre'].value_counts())
#%%
balanced_data = pd.concat([
    x.sample(n=5000, random_state=42, replace=True)
    for _, x in df.groupby('genre')
]).reset_index(drop=True)

# Display the shape of the new balanced dataset
print(f"Balanced dataset shape: {balanced_data.shape}")
print(balanced_data['genre'].value_counts())
#%%
genre_counts = balanced_data['genre'].value_counts()

plt.figure(figsize=(16, 10))
genre_counts.plot(kind='bar')
plt.title('Genre Distribution', fontsize=16)
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Number of Movies', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#%%
balanced_data = balanced_data.drop(columns =['movie_name'])
#%%
df_cleaned = balanced_data.copy()

#%% Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X = df_cleaned.drop(columns=['rating'])
y = df_cleaned['rating']

label_encoder = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = label_encoder.fit_transform(X[col].astype(str))
one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
genre_one_hot = one_hot_encoder.fit_transform(X[['genre']])
genre_one_hot_df = pd.DataFrame(genre_one_hot, columns=one_hot_encoder.get_feature_names_out(['genre']))
X = X.drop(columns=['genre']).reset_index(drop=True)
X = pd.concat([X, genre_one_hot_df], axis=1)

#%%
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)
print("Standardized Features:")
print(X_standardized_df.head())

#%%
normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X)
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)
print("\nNormalized Features:")
print(X_normalized_df.head())



#%%

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

vif_standardized = calculate_vif(X_standardized_df)
print("VIF Before Removal (Standardized):")
print(vif_standardized)

# Remove features with VIF > 10
high_vif_features = vif_standardized[vif_standardized['VIF'] > 10]['Feature']
X_standardized_reduced = X_standardized_df.drop(columns=high_vif_features)

# Recalculate VIF after removal
vif_standardized_reduced = calculate_vif(X_standardized_reduced)
print("VIF After Removal (Standardized):")
print(vif_standardized_reduced)

# Step 2: Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_standardized_reduced)
explained_variance = pca.explained_variance_ratio_
print("PCA Explained Variance Ratio:", explained_variance)
print("PCA Shape After Reduction:", X_pca.shape)

# Step 3: Use Random Forest Feature Importance
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_standardized_reduced, y)  # Use reduced features
feature_importances = pd.DataFrame({
    'Feature': X_standardized_reduced.columns,
    'Importance': rf_regressor.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Random Forest Feature Importance:")
print(feature_importances)

# Keep top features based on RF importance
top_features = feature_importances.head(10)['Feature']  # Retain top 10 features
X_rf_selected = X_standardized_reduced[top_features]

# Step 4: Optionally Apply SVD
svd = TruncatedSVD(n_components=6, random_state=42)
X_svd = svd.fit_transform(X_standardized_reduced)
print("SVD Shape After Reduction:", X_svd.shape)

# Final Outputs
print("\nFinal Selected Features (VIF and RF Importance):", X_rf_selected.columns.tolist())
print("\nFinal PCA Components Shape:", X_pca.shape)
print("\nFinal SVD Components Shape:", X_svd.shape)

#%%
vif_normalized = calculate_vif(X_normalized_df)
print("\nVIF Before Removal (Normalized):")
print(vif_normalized)

high_vif_features = vif_normalized[vif_normalized['VIF'] > 10]['Feature']
X_normalized_reduced = X_normalized_df.drop(columns=high_vif_features)
vif_normalized_reduced = calculate_vif(X_normalized_reduced)
print("\nVIF After Removal (Normalized):")
print(vif_normalized_reduced)
#%%
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca_normalized = pca.fit_transform(X_normalized_reduced)
explained_variance_normalized = pca.explained_variance_ratio_
print("\nPCA Explained Variance Ratio (Normalized):", explained_variance_normalized)
print("PCA Shape After Reduction (Normalized):", X_pca_normalized.shape)
#%%
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_normalized_reduced, y)  # Use reduced features
feature_importances_normalized = pd.DataFrame({
    'Feature': X_normalized_reduced.columns,
    'Importance': rf_regressor.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nRandom Forest Feature Importance (Normalized):")
print(feature_importances_normalized)

top_features_normalized = feature_importances_normalized.head(10)['Feature']
X_rf_selected_normalized = X_normalized_reduced[top_features_normalized]

#%%
svd = TruncatedSVD(n_components=4, random_state=42)
X_svd_normalized = svd.fit_transform(X_normalized_reduced)
print("\nSVD Shape After Reduction (Normalized):", X_svd_normalized.shape)
print("\nFinal Selected Features (VIF and RF Importance, Normalized):", X_rf_selected_normalized.columns.tolist())
print("\nFinal PCA Components Shape (Normalized):", X_pca_normalized.shape)
print("\nFinal SVD Components Shape (Normalized):", X_svd_normalized.shape)


#%%
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy.stats import zscore


def detect_and_remove_outliers(method, data, **kwargs):

    if method == 'lof':
        n_neighbors = kwargs.get('n_neighbors', 20)
        contamination = kwargs.get('contamination', 0.05)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outlier_labels = lof.fit_predict(data)
        return data[outlier_labels == 1]

    elif method == 'isolation_forest':
        contamination = kwargs.get('contamination', 0.05)
        random_state = kwargs.get('random_state', 42)
        iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
        outlier_labels = iso_forest.fit_predict(data)
        return data[outlier_labels == 1]

    elif method == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(data)
        return data[cluster_labels != -1]  # Keep only non-outlier points

    elif method == 'zscore':
        threshold = kwargs.get('threshold', 3)
        z_scores = np.abs(zscore(data))
        outlier_mask = (z_scores < threshold).all(axis=1)  # Identify rows where all features are below threshold
        return data[outlier_mask]

    else:
        raise ValueError("Invalid method. Choose 'lof', 'isolation_forest', 'dbscan', or 'zscore'.")



X_lof_cleaned = detect_and_remove_outliers('lof', X_rf_selected, n_neighbors=20, contamination=0.05)
print(f"Shape after LOF outlier removal: {X_lof_cleaned.shape}")

X_iso_cleaned = detect_and_remove_outliers('isolation_forest', X_rf_selected, contamination=0.05, random_state=42)
print(f"Shape after Isolation Forest outlier removal: {X_iso_cleaned.shape}")

X_dbscan_cleaned = detect_and_remove_outliers('dbscan', X_rf_selected, eps=0.5, min_samples=5)
print(f"Shape after DBSCAN outlier removal: {X_dbscan_cleaned.shape}")

X_zscore_cleaned = detect_and_remove_outliers('zscore', X_rf_selected, threshold=3)
print(f"Shape after Z-score outlier removal: {X_zscore_cleaned.shape}")

#%%
X_lof_cleaned_normalized = detect_and_remove_outliers('lof', X_rf_selected_normalized, n_neighbors=20, contamination=0.05)
print(f"Shape after LOF outlier removal (Normalized): {X_lof_cleaned_normalized.shape}")
X_iso_cleaned_normalized = detect_and_remove_outliers('isolation_forest', X_rf_selected_normalized, contamination=0.05, random_state=42)
print(f"Shape after Isolation Forest outlier removal (Normalized): {X_iso_cleaned_normalized.shape}")
X_dbscan_cleaned_normalized = detect_and_remove_outliers('dbscan', X_rf_selected_normalized, eps=0.5, min_samples=5)
print(f"Shape after DBSCAN outlier removal (Normalized): {X_dbscan_cleaned_normalized.shape}")

X_zscore_cleaned_normalized = detect_and_remove_outliers('zscore', X_rf_selected_normalized, threshold=3)
print(f"Shape after Z-score outlier removal (Normalized): {X_zscore_cleaned_normalized.shape}")


#%%
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso_forest.fit_predict(X_rf_selected)  # Use your selected features
X_cleaned_iso = X_rf_selected[outlier_labels == 1]
y_cleaned_iso = y[outlier_labels == 1]

print(f"Shape after Isolation Forest: {X_cleaned_iso.shape}")
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned_iso, y_cleaned_iso, test_size=0.2, random_state=42
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

#%% Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train_sm).fit()

y_pred_train = model.predict(X_train_sm)
y_pred_test = model.predict(X_test_sm)

mse = mean_squared_error(y_test, y_pred_test)
r_squared = r2_score(y_test, y_pred_test)
adjusted_r_squared = 1 - (1 - r_squared) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
aic = model.aic
bic = model.bic

evaluation_table = pd.DataFrame({
    "Metric": ["R-squared", "Adjusted R-squared", "AIC", "BIC", "MSE"],
    "Value": [r_squared, adjusted_r_squared, aic, bic, mse]
})
print(evaluation_table)

#%% T-test and F-test
print("T-Test Results:")
print(model.t_test(np.identity(len(model.params))))

print("\nF-Test Results:")
print(model.f_test(np.identity(len(model.params))))

#%% Confidence Interval Analysis
confidence_intervals = model.conf_int()
print("\nConfidence Intervals:")
print(confidence_intervals)

#%% Stepwise Regression
X_train_reset = pd.DataFrame(X_train).reset_index(drop=True)
y_train_reset = pd.Series(y_train).reset_index(drop=True)

def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_col in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_col]])).fit()
            new_pval[new_col] = model.pvalues[new_col]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f"Add {best_feature} with p-value {best_pval}")

        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f"Drop {worst_feature} with p-value {worst_pval}")

        if not changed:
            break
    return included
selected_features = stepwise_selection(X_train_reset, y_train_reset)

print("\nSelected Features via Stepwise Regression:")
print(selected_features)

#%%
print(model.summary())
#%% Final Model Visualization
plt.figure(figsize=(12, 8))
plt.plot(y_train.values, label='Train (Actual)', alpha=0.7)
plt.plot(y_pred_train, label='Train (Predicted)', linestyle='--')
plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test.values, label='Test (Actual)', alpha=0.7)
plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_pred_test, label='Test (Predicted)', linestyle='--')
plt.title('Train, Test, and Predicted Variables')
plt.xlabel('Observations')
plt.ylabel('Dependent Variable')
plt.legend()
plt.grid()
plt.show()
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.scatter(range(len(y_train)), y_train, label='Train (Actual)', alpha=0.5, s=10, color='blue')
plt.scatter(range(len(y_train)), y_pred_train, label='Train (Predicted)', alpha=0.5, s=10, color='orange')

test_range = range(len(y_train), len(y_train) + len(y_test))
plt.scatter(test_range, y_test, label='Test (Actual)', alpha=0.5, s=10, color='green')
plt.scatter(test_range, y_pred_test, label='Test (Predicted)', alpha=0.5, s=10, color='red')
plt.title('Train, Test, and Predicted Variables', fontsize=16)
plt.xlabel('Observations', fontsize=14)
plt.ylabel('Dependent Variable', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.7)
plt.tight_layout()
plt.show()

