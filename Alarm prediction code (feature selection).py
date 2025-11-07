# ====================== ALARM PREDICTION CODE: FEATURE SELECTION ======================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# === Uploading data ===
df = pd.read_csv(
    r"raw_dataset.csv",
    delimiter=",",
    encoding="utf-8",
    engine="python"
)
df.columns = df.columns.str.strip()

# === Cleaning and normalization ===
df['type'] = df['type'].astype(str).str.strip().str.lower()
print("Unique values in 'type':", df['type'].unique())

# Filter for relevant values
df = df[df['type'].isin(['normal', 'alarm', 'override'])]

# Map: normal and override = 0, alarm = 1
df['type'] = df['type'].map({'normal': 0, 'alarm': 1, 'override': 0})

# === Security check ===
if df.empty:
    raise ValueError("⚠️ No rows with type = 0 or 1 found. Check the data")

# === Balanced subsampling ===
df_sample = df.groupby('type', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 25000)))

# === 5. Cleaning unnecessary columns ===
df_sample = df_sample.drop(columns=['hash', 'id', 'time'], errors='ignore')

# === X and y separation ===
X = df_sample.drop(columns='type')
y = df_sample['type']

# === Saving ===
df_sample.to_csv(r"sample_feature_selection_ready.csv", index=False)

# === Column identification ===
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

# === Preprocessing pipeline with Imputer ===
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# === Feature selection pipeline ===
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('var_thresh', VarianceThreshold(threshold=0.0)),
    ('select_kbest', SelectKBest(mutual_info_classif, k=85))
])

# === Fit pipeline ===
X_transformed = pipeline.fit_transform(X, y)

# === Columns selected ===
encoded_cols = pipeline.named_steps['preprocessing'].get_feature_names_out()
var_kept = encoded_cols[pipeline.named_steps['var_thresh'].get_support()]
kbest_mask = pipeline.named_steps['select_kbest'].get_support()
kbest_selected = var_kept[kbest_mask]

print("\n✅ Top 35 features selected by SelectKBest:")
for f in kbest_selected:
    print("-", f)

# === Analysis and graph mutual_info_classif ===
import matplotlib.pyplot as plt

# Mutual information scores calculated by SelectKBest
mi_scores = pipeline.named_steps['select_kbest'].scores_

# Calculate the features remaining after variance
columns_after_var = var_kept

# It only takes the 35 selected ones
selected_scores = mi_scores[kbest_mask]
selected_features = columns_after_var[kbest_mask]

# Sort by descending score
sorted_pairs = sorted(zip(selected_features, selected_scores), key=lambda x: x[1], reverse=True)
features_sorted, scores_sorted = zip(*sorted_pairs)

# Plot of scores
plt.figure(figsize=(12, 6))
plt.barh(features_sorted, scores_sorted)
plt.xlabel("Mutual Information Score")
plt.title("Mutual Information (SelectKBest - Top 35 Feature)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# === Printing of selected features with MI scores ===
print("\n📋 Features selected by SelectKBest (ordered by MI importance):\n")
for i, (f, s) in enumerate(zip(features_sorted, scores_sorted), 1):
    print(f"{i:>2}. {f:60s} -> MI score: {s:.4f}")

# Statistics on information features
informative_count = sum(score > 0.01 for score in scores_sorted)
print(f"\n🔍 Feature with MI > 0.01: {informative_count} on 35")

# === Removal of redundant features among the 35 selected ===
print("\n🔁 Removal of closely related features (> 0.9)...")

# Create DataFrame with the 35 selected features
X_selected_df = pd.DataFrame(X_transformed, columns=kbest_selected)

# Calculate absolute correlation matrix
corr_matrix = X_selected_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features to remove
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X_deduplicated = X_selected_df.drop(columns=to_drop)

# Add the target column for saving
X_deduplicated['type'] = y.values
X_deduplicated.to_csv(r"dataset_filtered.csv", index=False)

print(f"\n✅ Final features after de-correlation: {X_deduplicated.shape[1]-1} (rimosse: {len(to_drop)})")