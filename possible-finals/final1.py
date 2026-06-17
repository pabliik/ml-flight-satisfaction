# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ================Data Exploration================

# %%
df = pd.read_csv('data/airline_passenger_satisfaction.csv')

# %%
#Shape of the dataset
print(f"Shape: {df.shape}")

# %%
#Dataset info
print("Data types")
df.dtypes

# %%
df.describe()

# %%
df.head()

# %%
#Missing values
print("Missing values")
print(df.isna().sum())
#Arrival Delay - 393 missing values.
#It is only 0.3% of whole dataset, so it can be dropped.

# %%
#Target distribution
print("Target distribution")
print(df['Satisfaction'].value_counts())

# %% [markdown]
# ================Data Preprocessing================

# %%
df = df.drop(columns=['ID']) # ID is not useful for prediction, so it can be dropped.
rows = df[df['Arrival Delay'].isna()].index
df = df.drop(index=rows) # Drop rows with missing values in 'Arrival Delay' column.

# %%
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
LE = LabelEncoder()
OE = OrdinalEncoder(categories=[["Economy", "Economy Plus", "Business"]])

df["Class_encoded"] = OE.fit_transform(df[["Class"]])
df["Class_encoded"] = df["Class_encoded"].astype(int)


label_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Satisfaction']
# one_hot_cols = ['Class']
for col in label_cols:
    df[col] = LE.fit_transform(df[col])

df = df.drop(columns=["Class"])
   

# %%
print(df.head())
print(df.info())

# %%
plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# %% [markdown]
# Departure Delay and Arrival Delay are highly correlated, so we can drop one of them. In this case, we will drop Departure Delay, because Arrival Delay is more informative for our analysis.

# %%
df = df.drop(columns=['Departure Delay'])

# %%
#Spliting data in training and testing
from sklearn.model_selection import train_test_split

X = df.drop(columns='Satisfaction')
y = df['Satisfaction']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_cols = ['Age', 'Flight Distance', 'Arrival Delay']
scale_columns = [col for col in df.columns if df[col].max() == 5]
numerical_cols += scale_columns
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
df[numerical_cols] = scaler.transform(df[numerical_cols])
X = df.drop(columns='Satisfaction')
print(df.describe())

# %% [markdown]
# ========Logistic Regression======

# %%
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train,y_train)
lr_prediction = lr.predict(X_test)


# %% [markdown]
# ===================Classification report for Logistic Regression===================

# %%
from sklearn.metrics import classification_report

lr_report = classification_report(y_test,lr_prediction)
print(lr_report)

# %%
from sklearn.model_selection import cross_validate

lr_scores = cross_validate(lr, X_train,y_train, scoring='f1', cv=5)
lr_final = lr_scores['test_score'].mean()
print(lr_final)

# %% [markdown]
# ==========Decision Tree==========

# %%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

# hyper parameter tuning
# random distribution
param_dist = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 3, 5, 10, 15, 20, 30],
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 4, 8, 16],
    'max_features': [None, 'sqrt', 'log2']
}

dtree_classifier = DecisionTreeClassifier(random_state=1)
random_search = RandomizedSearchCV(dtree_classifier, param_distributions=param_dist, 
                                   n_iter=100, cv=5, n_jobs=-1, scoring='f1', random_state=1, )

random_search.fit(X_train, y_train)
best_params_random = random_search.best_params_
best_score_random = random_search.best_score_
print(f"Best Parameters (Random Search): {best_params_random}")

# %%
best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)
tree_scores = cross_validate(best_model, X_train,y_train, scoring='f1', cv=5)
tree_final = tree_scores['test_score'].mean()

tree_report = classification_report(y_test, y_pred)
print(tree_report)

# %% [markdown]
# =============KNN=============

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Hyperparameter tuning for KNN
# Finding the best value for neighbors

k_values = range(7, 22, 2)
scores= []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', p=1)
    score = cross_val_score(knn, X_train, y_train, cv=5, scoring="f1").mean()
    scores.append(score)

best_k = k_values[np.argmax(scores)]
print(f"Best K: {best_k}, F1: {max(scores):.4f}")


# %%

clf = KNeighborsClassifier(n_neighbors=best_k, weights="distance", p=1)
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %%
# Evaluating all three models

knn_scores = cross_validate(clf, X_train,y_train, scoring='f1', cv=5)

print("LR:", lr_scores['test_score'].mean())
print("DT:", tree_scores['test_score'].mean())
print("KNN:", knn_scores['test_score'].mean())

# %%
import scipy.stats as stats
# t-test to test if decision tree performs significantly differently to the logistic regression model

stat1, p1 = stats.ttest_rel(tree_scores['test_score'], lr_scores['test_score'])

print("p-value:", p1)

# %%
# t-test to test if decision tree performs significantly differently to the knn model
stat2, p2 = stats.ttest_rel(tree_scores['test_score'], knn_scores['test_score'])

print("p-value:", p2)


