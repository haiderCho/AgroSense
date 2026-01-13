# %% [markdown]
# # Unified Crop Recommendation Analysis Pipeline
# 
# This script performs comprehensive EDA on the Crop Recommendation dataset.
# Each section is a separate cell for easy execution in VS Code Interactive / Jupyter.
#
# **Dataset:** `data/raw/Crop_recommendation.csv`  
# **Output:** `output/plots/`

# %% [markdown]
# ## Setup & Configuration

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Paths - works from both project root and notebooks folder
if os.path.exists(os.path.join("data", "raw", "Crop_recommendation.csv")):
    DATA_PATH = os.path.join("data", "raw", "Crop_recommendation.csv")
    OUTPUT_DIR = os.path.join("output", "plots")
else:
    DATA_PATH = os.path.join("..", "data", "raw", "Crop_recommendation.csv")
    OUTPUT_DIR = os.path.join("..", "output", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Archivo', 'Roboto', 'Arial']
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
PALETTE = sns.color_palette("husl", 8)
sns.set_palette(PALETTE)

print("Setup complete.")

# %% [markdown]
# ## Load Data

# %%
df = pd.read_csv(DATA_PATH)
print(f"Data loaded. Shape: {df.shape}")
print(df.head())

# %%
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target_col = 'label'
X = df.drop(target_col, axis=1)
y = df[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Numerical columns: {numerical_cols}")
print(f"Target: {target_col} ({y.nunique()} classes)")

# %% [markdown]
# ---
# ## Step 00A: Duplicates Check

# %%
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")
if duplicates > 0:
    print("Warning: Consider removing duplicates before training.")
else:
    print("No duplicates found.")

# %% [markdown]
# ---
# ## Step 00B: Skewness & Kurtosis Report

# %%
from scipy.stats import skew, kurtosis

skew_kurt = pd.DataFrame({
    'Feature': numerical_cols,
    'Skewness': [skew(df[col]) for col in numerical_cols],
    'Kurtosis': [kurtosis(df[col]) for col in numerical_cols]
})
print("Skewness & Kurtosis Report:")
print(skew_kurt.to_string(index=False))
print("\nInterpretation:")
print("  Skewness: |value| > 1 = highly skewed, may need log/sqrt transform")
print("  Kurtosis: |value| > 3 = heavy tails, may have outliers")

# %% [markdown]
# ---
# ## Step 00C: Variance Inflation Factor (VIF)

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("Variance Inflation Factor (VIF):")
print(vif_data.to_string(index=False))
print("\nInterpretation: VIF > 10 indicates severe multicollinearity.")

# %% [markdown]
# ---
# ## Step 00D: Statistical Tests (ANOVA / Kruskal-Wallis)

# %%
from scipy.stats import f_oneway, kruskal

print("Statistical Tests: Do features differ significantly across crop types?\n")
print(f"{'Feature':<15} {'ANOVA F-stat':<15} {'ANOVA p-value':<15} {'Kruskal H-stat':<15} {'Kruskal p-value':<15}")
print("-" * 75)

groups_by_label = [df[df[target_col] == label] for label in y.unique()]

for col in numerical_cols:
    # ANOVA (assumes normality)
    f_stat, p_anova = f_oneway(*[g[col] for g in groups_by_label])
    # Kruskal-Wallis (non-parametric)
    h_stat, p_kruskal = kruskal(*[g[col] for g in groups_by_label])
    
    print(f"{col:<15} {f_stat:<15.2f} {p_anova:<15.2e} {h_stat:<15.2f} {p_kruskal:<15.2e}")

print("\nInterpretation: p-value < 0.05 means the feature differs significantly across crop types.")

# %% [markdown]
# ---
# ## Step 01: Univariate Distributions

# %%
for col in numerical_cols:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x=col, kde=True, color=PALETTE[0], line_kws={'linewidth': 2})
    plt.title(f'Distribution of {col}', fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, f"01_dist_{col}.png"), dpi=300, bbox_inches='tight')
    plt.close()
print("Step 01 complete: Distributions saved.")

# %% [markdown]
# ---
# ## Step 02: Outlier Analysis (Boxplots)

# %%
# Summary boxplot
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[numerical_cols], palette="Set2")
plt.title('Box Plot of Numerical Features', fontweight='bold')
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_DIR, "02_boxplot_summary.png"), dpi=300, bbox_inches='tight')
plt.close()

# Individual boxplots
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], color=PALETTE[2])
    plt.title(f'Box Plot of {col}', fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, f"02_boxplot_{col}.png"), dpi=300, bbox_inches='tight')
    plt.close()
print("Step 02 complete: Boxplots saved.")

# %% [markdown]
# ---
# ## Step 03: Target Class Balance

# %%
plt.figure(figsize=(15, 8))
counts = y.value_counts()
sns.barplot(x=counts.index, y=counts.values, palette="viridis")
plt.title('Class Distribution', fontweight='bold')
plt.xlabel('Crop Label')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.savefig(os.path.join(OUTPUT_DIR, "03_class_balance.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Step 03 complete: Class balance saved.")
print(counts)

# %% [markdown]
# ---
# ## Step 04: Correlation Matrix

# %%
plt.figure(figsize=(12, 10))
corr_matrix = df[numerical_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix', fontweight='bold', pad=20)
plt.savefig(os.path.join(OUTPUT_DIR, "04_correlation_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Step 04 complete: Correlation matrix saved.")

# %% [markdown]
# ---
# ## Step 05: Feature vs Target (Boxen Plots)

# %%
for col in numerical_cols:
    plt.figure(figsize=(18, 6))
    sns.boxenplot(data=df, x=target_col, y=col, palette="cubehelix")
    plt.title(f'{col} vs Crop Label', fontweight='bold')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(OUTPUT_DIR, f"05_{col}_vs_target.png"), dpi=300, bbox_inches='tight')
    plt.close()
print("Step 05 complete: Feature vs Target plots saved.")

# %% [markdown]
# ---
# ## Step 06: Pairplot

# %%
subset = ['N', 'P', 'K', 'temperature', 'rainfall', 'label']
sns.pairplot(df[subset], hue='label', palette="bright", corner=True)
plt.savefig(os.path.join(OUTPUT_DIR, "06_pairplot_subset.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Step 06 complete: Pairplot saved.")

# %% [markdown]
# ---
# ## Step 07: PCA (Dimensionality Reduction)

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(14, 10))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="tab20", s=60, alpha=0.8)
plt.title(f'PCA Projection (Explained Var: {pca.explained_variance_ratio_.sum():.2f})', fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')
plt.savefig(os.path.join(OUTPUT_DIR, "07_pca_2d.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Step 07 complete: PCA saved.")

# %% [markdown]
# ---
# ## Step 08: t-SNE (Manifold Learning)

# %%
tsne = TSNE(n_components=2, verbose=1, perplexity=40, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(14, 10))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="tab20", s=60, alpha=0.8)
plt.title('t-SNE Visualization', fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')
plt.savefig(os.path.join(OUTPUT_DIR, "08_tsne_2d.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Step 08 complete: t-SNE saved.")

# %% [markdown]
# ---
# ## Step 09: K-Means Clustering

# %%
n_clusters = y.nunique()
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

ari = adjusted_rand_score(y, clusters)
print(f"K-Means Adjusted Rand Index: {ari:.4f}")

# Visualize on PCA
plt.figure(figsize=(14, 10))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="tab20", s=60, alpha=0.8, legend='full')
plt.title(f'K-Means Clusters (k={n_clusters}, ARI={ari:.2f})', fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster ID", ncol=2)
plt.savefig(os.path.join(OUTPUT_DIR, "09_kmeans_clusters.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Step 09 complete: K-Means saved.")

# %% [markdown]
# ---
# ## Step 10: Feature Importance (Random Forest)

# %%
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=importances, x='Importance', y='Feature', palette='viridis')
plt.title('Random Forest Feature Importance', fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, "10_feature_importance.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Step 10 complete: Feature Importance saved.")
print(importances)

# %% [markdown]
# ---
# ## Summary
# 
# All analysis steps complete. Plots saved to `output/plots/` with numbered prefixes:
# - `01_*`: Distributions
# - `02_*`: Boxplots
# - `03_*`: Class Balance
# - `04_*`: Correlation
# - `05_*`: Feature vs Target
# - `06_*`: Pairplot
# - `07_*`: PCA
# - `08_*`: t-SNE
# - `09_*`: K-Means
# - `10_*`: Feature Importance
