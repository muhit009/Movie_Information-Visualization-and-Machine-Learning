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
from sklearn.preprocessing import LabelEncoder

y = df_cleaned['genre']
X = df_cleaned.drop(columns=['genre'])

label_encoder = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = label_encoder.fit_transform(X[col].astype(str))
y = label_encoder.fit_transform(y)

#%%
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)
print("Standardized Features:")
print(X_standardized_df.head())
#%%
cov_matrix = np.cov(X_standardized_df, rowvar=False)
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', cbar=True)
plt.title('Covariance Matrix (Standardized Data)', fontsize=16)
plt.show()
#%%
corr_matrix = X_standardized_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix (Standardized Data)', fontsize=16)
plt.show()
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

high_vif_features = vif_standardized[vif_standardized['VIF'] > 10]['Feature']
X_standardized_reduced = X_standardized_df.drop(columns=high_vif_features)

# Recalculate VIF after removal
vif_standardized_reduced = calculate_vif(X_standardized_reduced)
print("VIF After Removal (Standardized):")
print(vif_standardized_reduced)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_standardized_reduced)
explained_variance = pca.explained_variance_ratio_
print("PCA Explained Variance Ratio:", explained_variance)
print("PCA Shape After Reduction:", X_pca.shape)
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_standardized_reduced, y)
feature_importances = pd.DataFrame({
    'Feature': X_standardized_reduced.columns,
    'Importance': rf_regressor.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Random Forest Feature Importance:")
print(feature_importances)

top_features = feature_importances.head(10)['Feature']
X_rf_selected = X_standardized_reduced[top_features]

svd = TruncatedSVD(n_components=6, random_state=42)
X_svd = svd.fit_transform(X_standardized_reduced)
print("SVD Shape After Reduction:", X_svd.shape)

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

pca = PCA(n_components=0.95)
X_pca_normalized = pca.fit_transform(X_normalized_reduced)
explained_variance_normalized = pca.explained_variance_ratio_
print("\nPCA Explained Variance Ratio (Normalized):", explained_variance_normalized)
print("PCA Shape After Reduction (Normalized):", X_pca_normalized.shape)

rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_normalized_reduced, y)
feature_importances_normalized = pd.DataFrame({
    'Feature': X_normalized_reduced.columns,
    'Importance': rf_regressor.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nRandom Forest Feature Importance (Normalized):")
print(feature_importances_normalized)

top_features_normalized = feature_importances_normalized.head(10)['Feature']
X_rf_selected_normalized = X_normalized_reduced[top_features_normalized]

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
#%% Train test Split for DT
y_zscore_cleaned = y[X_rf_selected.index]
y_zscore_cleaned = y_zscore_cleaned[:len(X_zscore_cleaned)]
X_train, X_test, y_train, y_test = train_test_split(
    X_zscore_cleaned,
    y_zscore_cleaned,
    test_size=0.2,
    random_state=42,
    stratify=y_zscore_cleaned
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
y_train_series = pd.Series(y_train)
y_test_series = pd.Series(y_test)

print("\nClass distribution in training set:")
print(y_train_series.value_counts(normalize=True))

print("\nClass distribution in test set:")
print(y_test_series.value_counts(normalize=True))
#%%  Baseline DT model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

baseline_tree = DecisionTreeClassifier(random_state=42)
baseline_tree.fit(X_train, y_train)

y_pred = baseline_tree.predict(X_test)
print(f"Baseline Accuracy: {accuracy_score(y_test, y_pred)}")

#%% Pre-Pruning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_tree = grid_search.best_estimator_
print(f"Best Parameters: {best_params}")
#%%
from sklearn.metrics import classification_report, accuracy_score
pre_pruned_tree = DecisionTreeClassifier(
    random_state=42,
    criterion=best_params['criterion'],
    splitter=best_params['splitter'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features']
)

pre_pruned_tree.fit(X_train, y_train)

y_pred = pre_pruned_tree.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
#%%
from sklearn.tree import DecisionTreeClassifier
X_t = X_train[:1000]
y_t = y_train[:1000]
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_t,y_t)
ccp_alphas = path.ccp_alphas
impurities = path.impurities
ccp_alphas = ccp_alphas[ccp_alphas >= 0]
#%%
trees = []
for alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    tree.fit(X_train, y_train)
    trees.append(tree)


#%%
from sklearn.metrics import accuracy_score

train_scores = [accuracy_score(y_train, tree.predict(X_train)) for tree in trees]
test_scores = [accuracy_score(y_test, tree.predict(X_test)) for tree in trees]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, label="Train Accuracy", marker='o')
plt.plot(ccp_alphas, test_scores, label="Test Accuracy", marker='o')
plt.xlabel("ccp_alpha")
plt.ylabel("Accuracy")
plt.title("Effect of ccp_alpha on Model Performance")
plt.legend()
plt.grid(True)
plt.show()

#%%
optimal_ccp_alpha = ccp_alphas[np.argmax(test_scores)]
print(f"Optimal ccp_alpha: {optimal_ccp_alpha}")

#%%
post_pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=optimal_ccp_alpha)
post_pruned_tree.fit(X_train, y_train)
#%%
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import numpy as np

#%%
def calculate_specificity_multiclass(y_true, y_pred, average='macro'):
    cm = confusion_matrix(y_true, y_pred)
    specificity_per_class = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    if average == 'macro':
        return np.mean(specificity_per_class)
    elif average == 'weighted':
        weights = np.sum(cm, axis=1) / np.sum(cm)
        return np.sum(specificity_per_class * weights)

#%%
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)


    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)


    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    specificity = calculate_specificity_multiclass(y_test, y_pred, average='macro')

    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"Specificity (Macro): {specificity:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")


    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

    print(f"AUC (Macro): {roc_auc:.4f}")
#%%
evaluate_model(pre_pruned_tree, X_test, y_test)
#%%
evaluate_model(post_pruned_tree, X_test, y_test)

#%%
from sklearn.preprocessing import label_binarize
y_prob_pre_pruned = pre_pruned_tree.predict_proba(X_test)
y_prob_post_pruned = post_pruned_tree.predict_proba(X_test)

classes = np.unique(y_test)
y_test_binarized = label_binarize(y_test, classes=classes)

fpr_pre, tpr_pre, roc_auc_pre = {}, {}, {}
for i in range(len(classes)):
    fpr_pre[i], tpr_pre[i], _ = roc_curve(y_test_binarized[:, i], y_prob_pre_pruned[:, i])
    roc_auc_pre[i] = auc(fpr_pre[i], tpr_pre[i])

fpr_post, tpr_post, roc_auc_post = {}, {}, {}
for i in range(len(classes)):
    fpr_post[i], tpr_post[i], _ = roc_curve(y_test_binarized[:, i], y_prob_post_pruned[:, i])
    roc_auc_post[i] = auc(fpr_post[i], tpr_post[i])

# Plot ROC-AUC Curve for Pre-Pruned Tree
plt.figure(figsize=(12, 8))
for i in range(len(classes)):
    plt.plot(fpr_pre[i], tpr_pre[i], label=f'Class {classes[i]} (AUC = {roc_auc_pre[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Pre-Pruned Tree (Multi-Class)')
plt.legend(loc='best')
plt.grid()
plt.show()

# Plot ROC-AUC Curve for Post-Pruned Tree
plt.figure(figsize=(12, 8))
for i in range(len(classes)):
    plt.plot(fpr_post[i], tpr_post[i], label=f'Class {classes[i]} (AUC = {roc_auc_post[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Post-Pruned Tree (Multi-Class)')
plt.legend(loc='best')
plt.grid()
plt.show()
#%%
def perform_stratified_kfold(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"Stratified K-Fold Cross-Validation Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

print("\nPre-Pruned Tree Stratified K-Fold Cross-Validation:")
perform_stratified_kfold(pre_pruned_tree, X_train, y_train)

print("\nPost-Pruned Tree Stratified K-Fold Cross-Validation:")
perform_stratified_kfold(post_pruned_tree, X_train, y_train)
#%% Logistic Regression
y_zscore_cleaned = y[X_rf_selected.index]
y_zscore_cleaned = y_zscore_cleaned[:len(X_zscore_cleaned)]
X_train, X_test, y_train, y_test = train_test_split(
    X_zscore_cleaned,
    y_zscore_cleaned,
    test_size=0.2,
    random_state=42,
    stratify=y_zscore_cleaned
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import numpy as np

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

#%%
print(y_pred)

#%% Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

if cm.shape == (2, 2):
    TN, FP, FN, TP = cm.ravel()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


    print("Metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"F1-Score: {f1_score:.2f}")

else:
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

#%%
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

#%%
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
classes = np.unique(y_test)  # Extract the unique classes
y_test_binarized = label_binarize(y_test, classes=classes)
y_pred_prob = log_reg.predict_proba(X_test)
classes = np.unique(y_test)  # Extract the unique classes
y_test_binarized = label_binarize(y_test, classes=classes)
y_pred_prob = log_reg.predict_proba(X_test)
fpr = {}
tpr = {}
roc_auc = {}

for i, class_label in enumerate(classes):
    fpr[class_label], tpr[class_label], _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
    roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])

fpr_macro, tpr_macro, _ = roc_curve(y_test_binarized.ravel(), y_pred_prob.ravel())
roc_auc_macro = auc(fpr_macro, tpr_macro)
plt.figure(figsize=(10, 8))
for class_label in classes:
    plt.plot(fpr[class_label], tpr[class_label],
             label=f"Class {class_label} (AUC = {roc_auc[class_label]:.2f})")

plt.plot(fpr_macro, tpr_macro, label=f"Macro-Average (AUC = {roc_auc_macro:.2f})", linestyle='--', color='black')

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC-AUC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
roc_auc_macro = roc_auc_score(y_test_binarized, y_pred_prob, average="macro")
roc_auc_weighted = roc_auc_score(y_test_binarized, y_pred_prob, average="weighted")

print(f"Macro-Average AUC: {roc_auc_macro:.2f}")
print(f"Weighted-Average AUC: {roc_auc_weighted:.2f}")

#%%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#%%
from sklearn.metrics import make_scorer
log_reg = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
cv_accuracy = cross_val_score(log_reg, X_train, y_train, cv=skf, scoring='accuracy')

cv_f1_macro = cross_val_score(log_reg, X_train, y_train, cv=skf,
                               scoring=make_scorer(f1_score, average='macro'))


cv_roc_auc_weighted = cross_val_score(log_reg, X_train, y_train, cv=skf,
                                       scoring='roc_auc_ovr_weighted')

print("Stratified K-Fold Cross-Validation Results:")
print(f"Accuracy Scores: {cv_accuracy}")
print(f"Mean Accuracy: {cv_accuracy.mean():.2f}")
print(f"Standard Deviation: {cv_accuracy.std():.2f}\n")

print(f"F1 Macro Scores: {cv_f1_macro}")
print(f"Mean F1 Macro: {cv_f1_macro.mean():.2f}")
print(f"Standard Deviation: {cv_f1_macro.std():.2f}\n")

print(f"ROC-AUC Weighted Scores: {cv_roc_auc_weighted}")
print(f"Mean ROC-AUC Weighted: {cv_roc_auc_weighted.mean():.2f}")
print(f"Standard Deviation: {cv_roc_auc_weighted.std():.2f}\n")

#%% KNN
y_lof_cleaned_normalized = y[:len(X_lof_cleaned_normalized)]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_lof_cleaned_normalized,
    y_lof_cleaned_normalized,
    test_size=0.2,
    random_state=42,
    stratify=y_lof_cleaned_normalized
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

#%%  Elbow Method
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

k_values = range(1, 50)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Elbow Method for Finding Optimal K')
plt.grid(True)
plt.xticks(k_values)
plt.show()

optimal_k = k_values[cv_scores.index(max(cv_scores))]
print(f"Optimal K: {optimal_k}")

#%% Model
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train, y_train)

accuracy = knn_optimal.score(X_test, y_test)
print(f"Test Set Accuracy with Optimal K ({optimal_k}): {accuracy:.2f}")

#%%
y_pred = knn_optimal.predict(X_test)
y_pred_prob = knn_optimal.predict_proba(X_test)[:, 1]

#%% Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

if cm.shape == (2, 2):  # Binary case
    TN, FP, FN, TP = cm.ravel()

    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("Metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"F1-Score: {f1_score:.2f}")

else:
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

#%%
if len(np.unique(y_test)) == 2:  # Check for binary classification
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC and AUC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    print(f"ROC-AUC: {roc_auc:.2f}")
else:
    print("ROC-AUC is not available for multi-class classification without binarization.")

if len(np.unique(y_test)) > 2:  # Multi-class case
    from sklearn.preprocessing import label_binarize

    # Binarize the labels
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
    y_pred_prob_multi = knn_optimal.predict_proba(X_test)

    # Calculate AUC for each class
    roc_auc_multi = roc_auc_score(y_test_binarized, y_pred_prob_multi, average="macro")
    print(f"Macro-Averaged ROC-AUC: {roc_auc_multi:.2f}")
#%%
classes = np.unique(y_test)  # Extract unique classes
y_test_binarized = label_binarize(y_test, classes=classes)
y_pred_prob = knn_optimal.predict_proba(X_test)  # Probabilities for each class

# Compute ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}

for i, class_label in enumerate(classes):
    fpr[class_label], tpr[class_label], _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
    roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])

# Plot ROC curves for all classes
plt.figure(figsize=(10, 8))
for class_label in classes:
    plt.plot(fpr[class_label], tpr[class_label],
             label=f"Class {class_label} (AUC = {roc_auc[class_label]:.2f})")

# Plot random guess line
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess", lw=2, color='black')

# Customize plot
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
#%%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation for accuracy
cv_scores_accuracy = cross_val_score(knn_optimal, X_train, y_train, cv=skf, scoring='accuracy')

# Print results
print("Stratified K-Fold Cross-Validation Results:")
print(f"Accuracy Scores: {cv_scores_accuracy}")
print(f"Mean Accuracy: {cv_scores_accuracy.mean():.2f}")
print(f"Standard Deviation: {cv_scores_accuracy.std():.2f}")

#%%
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

cv_f1 = cross_val_score(knn, X_train, y_train, cv=5, scoring=make_scorer(f1_score, average='macro'))
#%%
k_values = range(1, 50)
#%%
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
cv_scores = cross_val_score(knn, X_train, y_train, cv=rskf, scoring='accuracy')

#%% SVM
from sklearn.model_selection import train_test_split

# RBF Kernel: Normalized data
y_iso_cleaned_normalized = y[:len(X_iso_cleaned_normalized)]  # Ensure alignment
X_train_rbf, X_test_rbf, y_train_rbf, y_test_rbf = train_test_split(
    X_iso_cleaned_normalized, y_iso_cleaned_normalized, test_size=0.2, random_state=42, stratify=y_iso_cleaned_normalized
)

# Polynomial and Linear Kernels: Original data
y_iso_cleaned = y[:len(X_iso_cleaned)]  # Ensure alignment
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_iso_cleaned, y_iso_cleaned, test_size=0.2, random_state=42, stratify=y_iso_cleaned
)

# Use the same split for Linear kernel as Polynomial kernel
X_train_linear, X_test_linear, y_train_linear, y_test_linear = X_train_poly, X_test_poly, y_train_poly, y_test_poly

#%% RBF Kernel
from sklearn.svm import SVC

svm_rbf = SVC(kernel='rbf', random_state=42, probability=True)
svm_rbf.fit(X_train_rbf, y_train_rbf)
y_pred_rbf = svm_rbf.predict(X_test_rbf)
accuracy_rbf = svm_rbf.score(X_test_rbf, y_test_rbf)
print(f"RBF Kernel Accuracy: {accuracy_rbf:.2f}")

#%% Polynomial Kernel
svm_poly = SVC(kernel='poly', degree=3, random_state=42, probability=True)
svm_poly.fit(X_train_poly, y_train_poly)

# Predictions and Accuracy
y_pred_poly = svm_poly.predict(X_test_poly)
accuracy_poly = svm_poly.score(X_test_poly, y_test_poly)
print(f"Polynomial Kernel Accuracy: {accuracy_poly:.2f}")

#%% Linear Kernel
svm_linear = SVC(kernel='linear', random_state=42, probability=True)
svm_linear.fit(X_train_linear, y_train_linear)

# Predictions and Accuracy
y_pred_linear = svm_linear.predict(X_test_linear)
accuracy_linear = svm_linear.score(X_test_linear, y_test_linear)
print(f"Linear Kernel Accuracy: {accuracy_linear:.2f}")

#%%
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt

def evaluate_model(y_test, y_pred, model_name):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()

    # Classification Report
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

evaluate_model(y_test_rbf, y_pred_rbf, "RBF Kernel")

evaluate_model(y_test_poly, y_pred_poly, "Polynomial Kernel")

evaluate_model(y_test_linear, y_pred_linear, "Linear Kernel")

#%%
def plot_roc_auc_multiclass(model, X_test, y_test, model_name):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import numpy as np

    # Binarize labels for multi-class
    classes = np.unique(y_test)
    y_test_binarized = label_binarize(y_test, classes=classes)
    y_pred_prob = model.predict_proba(X_test)

    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i, class_label in enumerate(classes):
        fpr[class_label], tpr[class_label], _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
        roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])

    # Plot ROC curves for all classes
    plt.figure(figsize=(10, 8))
    for class_label in classes:
        plt.plot(fpr[class_label], tpr[class_label],
                 label=f"Class {class_label} (AUC = {roc_auc[class_label]:.2f})")

    # Plot random guess line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Random Guess")

    # Customize the plot
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multi-Class ROC Curve: {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Compute and print macro-averaged AUC
    macro_auc = roc_auc_score(y_test_binarized, y_pred_prob, average="macro")
    print(f"{model_name} Macro-Averaged ROC-AUC: {macro_auc:.2f}")
#%%
plot_roc_auc_multiclass(svm_rbf, X_test_rbf, y_test_rbf, "RBF Kernel")
plot_roc_auc_multiclass(svm_poly, X_test_poly, y_test_poly, "Polynomial Kernel")
plot_roc_auc_multiclass(svm_linear, X_test_linear, y_test_linear, "Linear Kernel")

#%%
def stratified_k_fold_validation(model, X, y, model_name, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    print(f"{model_name} Stratified K-Fold Results:")
    print(f"Fold Scores: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.2f}")
    print(f"Standard Deviation: {cv_scores.std():.2f}\n")
#%%
stratified_k_fold_validation(svm_rbf, X_iso_cleaned_normalized, y[:len(X_iso_cleaned_normalized)], "RBF Kernel")
stratified_k_fold_validation(svm_poly, X_iso_cleaned, y[:len(X_iso_cleaned)], "Polynomial Kernel")
stratified_k_fold_validation(svm_linear, X_iso_cleaned, y[:len(X_iso_cleaned)], "Linear Kernel")


#%% Naive Bayes
from sklearn.model_selection import train_test_split

y_lof_cleaned_normalized = y[:len(X_lof_cleaned_normalized)]

X_train, X_test, y_train, y_test = train_test_split(
    X_lof_cleaned_normalized,
    y_lof_cleaned_normalized,
    test_size=0.2,
    random_state=42,
    stratify=y_lof_cleaned_normalized
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
#%%
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
accuracy = nb_model.score(X_test, y_test)
print(f"Naive Bayes Model Accuracy: {accuracy:.2f}")

#%%
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))

#%%
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Binarize the labels for multi-class
classes = np.unique(y_test)
y_test_binarized = label_binarize(y_test, classes=classes)
y_pred_prob = nb_model.predict_proba(X_test)
fpr = {}
tpr = {}
roc_auc = {}

for i, class_label in enumerate(classes):
    fpr[class_label], tpr[class_label], _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
    roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
plt.figure(figsize=(10, 8))
for class_label in classes:
    plt.plot(fpr[class_label], tpr[class_label],
             label=f"Class {class_label} (AUC = {roc_auc[class_label]:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess", lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
#%%
from sklearn.metrics import roc_auc_score
macro_auc = roc_auc_score(y_test_binarized, y_pred_prob, average="macro")
print(f"Macro-Averaged ROC-AUC: {macro_auc:.2f}")
weighted_auc = roc_auc_score(y_test_binarized, y_pred_prob, average="weighted")
print(f"Weighted-Averaged ROC-AUC: {weighted_auc:.2f}")
#%%
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.naive_bayes import GaussianNB

def stratified_k_fold_naive_bayes(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Ensure X and y are NumPy arrays for proper indexing
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.values
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.values

    accuracy_scores = []
    auc_scores = []

    for train_idx, test_idx in skf.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        y_pred_prob = model.predict_proba(X_test)

        accuracy = model.score(X_test, y_test)
        accuracy_scores.append(accuracy)

        y_test_binarized = label_binarize(y_test, classes=np.unique(y))
        auc = roc_auc_score(y_test_binarized, y_pred_prob, average="macro")
        auc_scores.append(auc)

    print(f"K-Fold Accuracy Scores: {accuracy_scores}")
    print(f"Mean Accuracy: {np.mean(accuracy_scores):.2f}, Standard Deviation: {np.std(accuracy_scores):.2f}")
    print(f"K-Fold AUC Scores: {auc_scores}")
    print(f"Mean AUC: {np.mean(auc_scores):.2f}, Standard Deviation: {np.std(auc_scores):.2f}")
#%%
nb_model = GaussianNB()

stratified_k_fold_naive_bayes(nb_model, X_lof_cleaned_normalized, y[:len(X_lof_cleaned_normalized)], n_splits=5)



