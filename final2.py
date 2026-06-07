import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load and Clean Data
df = pd.read_csv('../data/airline_passenger_satisfaction.csv')
df = df.drop(columns=['ID'])
df = df.dropna(subset=['Arrival Delay'])

# 2. Preprocessing
OE = OrdinalEncoder(categories=[["Economy", "Economy Plus", "Business"]])
df["Class_encoded"] = OE.fit_transform(df[["Class"]]).astype(int)
df = df.drop(columns=["Class"])

LE = LabelEncoder()
label_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Satisfaction']
for col in label_cols:
    df[col] = LE.fit_transform(df[col])

df = df.drop(columns=['Departure Delay'])

# 3. Split
X = df.drop(columns='Satisfaction')
y = df['Satisfaction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify split
print(f"Training set shape: {X_train.shape}")
print(f"Testing set target distribution:\n{y_train.value_counts(normalize=True)}")

# 4. Scaling
scaler = StandardScaler()
numerical_cols = ['Age', 'Flight Distance', 'Arrival Delay']
scale_columns = [col for col in df.columns if df[col].max() == 5]
numerical_cols += scale_columns

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Verify scaling
print("Scaled Numerical Columns Stats (Mean ~0, Std ~1):")
print(X_train[numerical_cols].agg(['mean', 'std']).round(2))

# 5. Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_prediction = lr.predict(X_test)
lr_scores = cross_validate(lr, X_train, y_train, scoring='f1', cv=5)
lr_final = lr_scores['test_score'].mean()
print(f"\nLR F1 Score: {lr_final:.4f}")

# 6. Decision Tree
param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4]
}
dtree_classifier = DecisionTreeClassifier(random_state=1)
random_search = RandomizedSearchCV(dtree_classifier, param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1, scoring='f1')
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
tree_scores = cross_validate(best_model, X_train, y_train, scoring='f1', cv=5)
tree_final = tree_scores['test_score'].mean()
print(f"DT F1 Score: {tree_final:.4f}")

# 7. KNN
k_values = range(7, 15, 2)
scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k, weights='distance', p=1), X_train, y_train, cv=5, scoring="f1").mean() for k in k_values]
best_k = k_values[np.argmax(scores)]
clf = KNeighborsClassifier(n_neighbors=best_k, weights="distance", p=1)
clf.fit(X_train, y_train)
knn_scores = cross_validate(clf, X_train, y_train, scoring='f1', cv=5)
print(f"KNN F1 Score: {knn_scores['test_score'].mean():.4f}")

# 8. Comparison Table
results_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "KNN"],
    "Mean CV F1 Score": [lr_final, tree_final, knn_scores['test_score'].mean()]
})
print("\nFinal Model Comparison:")
print(results_df.sort_values(by="Mean CV F1 Score", ascending=False))

# 9. Statistical Significance
stat1, p1 = stats.ttest_rel(tree_scores['test_score'], lr_scores['test_score'])
print(f"\nT-test DT vs LR p-value: {p1:.4f}")