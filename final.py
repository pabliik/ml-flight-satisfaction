import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_validate, cross_val_score, RandomizedSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


# =============================================================================
# 1. DATA LOADING
# =============================================================================

df = pd.read_csv('data/airline_passenger_satisfaction.csv')

print("=" * 60)
print("DATA EXPLORATION")
print("=" * 60)

print(f"\nShape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nDescriptive statistics:\n{df.describe()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing values:\n{df.isna().sum()}")
# Arrival Delay has 393 missing values (~0.3% of dataset) — safe to drop rows.

print(f"\nTarget distribution:\n{df['Satisfaction'].value_counts()}")
print(f"\nTarget distribution (%):\n{df['Satisfaction'].value_counts(normalize=True).round(3)}")


# =============================================================================
# 2. DATA PREPROCESSING
# =============================================================================

print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Drop ID (not predictive) and rows with missing Arrival Delay
df = df.drop(columns=['ID'])
df = df.dropna(subset=['Arrival Delay'])

# Encode Class ordinally (Economy < Economy Plus < Business)
OE = OrdinalEncoder(categories=[["Economy", "Economy Plus", "Business"]])
df["Class_encoded"] = OE.fit_transform(df[["Class"]]).astype(int)
df = df.drop(columns=["Class"])

# Label-encode binary categorical columns and target
LE = LabelEncoder()
label_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Satisfaction']
for col in label_cols:
    df[col] = LE.fit_transform(df[col])

# Drop Departure Delay — highly correlated with Arrival Delay (multicollinearity)
df = df.drop(columns=['Departure Delay'])

print(f"\nDataset after preprocessing:\n{df.head()}")
print(f"\nColumn info:\n{df.info()}")


# =============================================================================
# 3. CORRELATION ANALYSIS
# =============================================================================

plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


# =============================================================================
# 4. TRAIN / TEST SPLIT
# =============================================================================

X = df.drop(columns='Satisfaction')
y = df['Satisfaction']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape:     {X_test.shape}")
print(f"\nTarget distribution in training set:\n{y_train.value_counts(normalize=True).round(3)}")


# =============================================================================
# 5. SCALING
# =============================================================================

scaler = StandardScaler()

# Continuous features scaled to remove unit bias (StandardScaler: mean=0, std=1).
# Survey rating columns (Likert scale 1–5) are also scaled: although ordinal,
# their ranges differ from binary-encoded columns (0/1) and Class_encoded (0–2),
# so scaling puts all features on a comparable magnitude — important for
# distance-based models (KNN) and regularised models (Logistic Regression).
# The max==5 filter reliably identifies all Likert columns in this dataset.
numerical_cols = ['Age', 'Flight Distance', 'Arrival Delay']
likert_cols = [col for col in X_train.columns if X_train[col].max() == 5]
cols_to_scale = numerical_cols + likert_cols

# Fit on train only — transform train and test separately to prevent leakage
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

print("\nScaled column stats on training set (mean ≈ 0, std ≈ 1):")
print(X_train[cols_to_scale].agg(['mean', 'std']).round(2))


# =============================================================================
# 6. LOGISTIC REGRESSION
# =============================================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_prediction = lr.predict(X_test)

print("\nClassification Report (Test Set):")
print(classification_report(y_test, lr_prediction))

lr_scores = cross_validate(lr, X_train, y_train, scoring='f1', cv=5)
lr_final = lr_scores['test_score'].mean()
print(f"Cross-validated F1 (5-fold): {lr_final:.4f}")


# =============================================================================
# 7. DECISION TREE
# =============================================================================

print("\n" + "=" * 60)
print("DECISION TREE")
print("=" * 60)

param_dist = {
    'criterion':          ['gini', 'entropy', 'log_loss'],
    'max_depth':          [None, 3, 5, 10, 15, 20, 30],
    'min_samples_split':  [2, 5, 10, 20, 50],
    'min_samples_leaf':   [1, 2, 4, 8, 16],
    'max_features':       [None, 'sqrt', 'log2'],
}

dt = DecisionTreeClassifier(random_state=1)
random_search = RandomizedSearchCV(
    dt, param_distributions=param_dist,
    n_iter=100, cv=5, n_jobs=-1, scoring='f1', random_state=1
)
random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")

best_dt = random_search.best_estimator_
dt_prediction = best_dt.predict(X_test)

print("\nClassification Report (Test Set):")
print(classification_report(y_test, dt_prediction))

tree_scores = cross_validate(best_dt, X_train, y_train, scoring='f1', cv=5)
tree_final = tree_scores['test_score'].mean()
print(f"Cross-validated F1 (5-fold): {tree_final:.4f}")


# =============================================================================
# 8. K-NEAREST NEIGHBOURS
# =============================================================================

print("\n" + "=" * 60)
print("K-NEAREST NEIGHBOURS")
print("=" * 60)

# Tune k using cross-validation on odd values to avoid ties
k_values = range(7, 22, 2)
k_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', p=1)
    score = cross_val_score(knn, X_train, y_train, cv=5, scoring='f1').mean()
    k_scores.append(score)

best_k = k_values[np.argmax(k_scores)]
print(f"Best k: {best_k}  |  CV F1: {max(k_scores):.4f}")

best_knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance', p=1)
best_knn.fit(X_train, y_train)
knn_prediction = best_knn.predict(X_test)

print(f"\nTest Set Accuracy: {accuracy_score(y_test, knn_prediction):.4f}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, knn_prediction))

knn_scores = cross_validate(best_knn, X_train, y_train, scoring='f1', cv=5)
knn_final = knn_scores['test_score'].mean()
print(f"Cross-validated F1 (5-fold): {knn_final:.4f}")


# =============================================================================
# 9. MODEL COMPARISON
# =============================================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results = pd.DataFrame({
    "Model":            ["Logistic Regression", "Decision Tree", "KNN"],
    "CV F1 (mean)":     [lr_final, tree_final, knn_final],
    "CV F1 (std)":      [
        lr_scores['test_score'].std(),
        tree_scores['test_score'].std(),
        knn_scores['test_score'].std(),
    ],
})
print(results.sort_values(by="CV F1 (mean)", ascending=False).to_string(index=False))


# =============================================================================
# 10. STATISTICAL SIGNIFICANCE (paired t-tests)
# =============================================================================

print("\n" + "=" * 60)
print("STATISTICAL SIGNIFICANCE")
print("=" * 60)

# Paired t-test compares the per-fold CV scores of two models directly.
# Because both models are evaluated on the same folds, the scores are paired —
# a paired t-test accounts for fold-to-fold variance and is more sensitive
# than an independent-samples test. Null hypothesis: the two models have equal
# mean F1. A p-value < 0.05 indicates the difference is statistically
# significant and unlikely due to random variation across folds.

stat1, p1 = stats.ttest_rel(tree_scores['test_score'], lr_scores['test_score'])
print(f"Decision Tree vs Logistic Regression  |  t={stat1:.4f}, p={p1:.4f}")
print(f"  → {'Significant difference' if p1 < 0.05 else 'No significant difference'} (α=0.05)")

stat2, p2 = stats.ttest_rel(tree_scores['test_score'], knn_scores['test_score'])
print(f"Decision Tree vs KNN                  |  t={stat2:.4f}, p={p2:.4f}")
print(f"  → {'Significant difference' if p2 < 0.05 else 'No significant difference'} (α=0.05)")