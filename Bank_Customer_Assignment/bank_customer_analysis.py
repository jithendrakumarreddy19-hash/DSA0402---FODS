# ============================================================
# DSA0402 - FUNDAMENTALS OF DATA SCIENCE
# Bank Customer Subscription Prediction and Customer Segmentation
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.cluster import KMeans

# ------------------------------------------------------------
# 1. DATA RETRIEVAL AND INSPECTION
# ------------------------------------------------------------

print("=" * 60)
print("1. DATASET INSPECTION")
print("=" * 60)

df = pd.read_csv("bank-full.csv", sep=";")

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nSUBSCRIPTION COUNT:")
print(df["y"].value_counts())

# ------------------------------------------------------------
# 2. DATA PREPROCESSING
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("2. DATA PREPROCESSING")
print("=" * 60)

# Convert target: yes = 1, no = 0
df["y"] = df["y"].map({"yes": 1, "no": 0})

# Remove duplicate rows
duplicates = df.duplicated().sum()
print("\nNUMBER OF DUPLICATE ROWS:", duplicates)

df = df.drop_duplicates()

print("\nDATASET AFTER PREPROCESSING:")
print(df.shape)

print("\nTARGET VARIABLE:")
print(df["y"].value_counts())

# ------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Subscription distribution
plt.figure(figsize=(6, 4))
df["y"].value_counts().plot(kind="bar")
plt.title("Customer Subscription Distribution")
plt.xlabel("Subscription (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Age distribution
plt.figure(figsize=(7, 4))
plt.hist(df["age"], bins=20)
plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Job distribution
plt.figure(figsize=(10, 5))
df["job"].value_counts().plot(kind="bar")
plt.title("Customer Distribution by Job")
plt.xlabel("Job")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Balance outlier analysis
plt.figure(figsize=(7, 4))
plt.boxplot(df["balance"])
plt.title("Customer Balance Distribution")
plt.ylabel("Balance")
plt.tight_layout()
plt.show()

# Correlation heatmap using numerical features
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))
plt.imshow(numeric_df.corr(), cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.xticks(
    range(len(numeric_df.columns)),
    numeric_df.columns,
    rotation=45,
    ha="right"
)
plt.yticks(
    range(len(numeric_df.columns)),
    numeric_df.columns
)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY")

# ------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("4. DESCRIPTIVE STATISTICS")
print("=" * 60)

stats_columns = ["age", "balance", "duration", "campaign"]

print("\nMEAN:")
print(df[stats_columns].mean())

print("\nVARIANCE:")
print(df[stats_columns].var())

print("\nCOVARIANCE:")
print(df[stats_columns].cov())

print("\nCORRELATION:")
print(df[["age", "balance", "duration", "campaign", "y"]].corr())

# ------------------------------------------------------------
# 5. STATISTICAL INFERENCE
# 95% CONFIDENCE INTERVAL FOR MEAN AGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("5. STATISTICAL INFERENCE")
print("=" * 60)

mean_age = df["age"].mean()
std_age = df["age"].std()
n = len(df)

confidence = 0.95
standard_error = std_age / (n ** 0.5)

margin_of_error = stats.t.ppf(
    (1 + confidence) / 2,
    n - 1
) * standard_error

lower_limit = mean_age - margin_of_error
upper_limit = mean_age + margin_of_error

print("\nMean Age:", mean_age)
print("\n95% Confidence Interval:")
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

print("\nSTATISTICAL INFERENCE COMPLETED")

# ------------------------------------------------------------
# 6. MACHINE LEARNING
# Logistic Regression, Decision Tree, kNN
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("6. MACHINE LEARNING")
print("=" * 60)

X = df.drop("y", axis=1)
y = df["y"]

categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}
confusion_matrices = {}

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = [accuracy, precision, recall, f1]
    confusion_matrices[name] = cm

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-Score :", f1)

    print("\nConfusion Matrix:")
    print(cm)

# Model comparison table
results_df = pd.DataFrame(
    results,
    index=["Accuracy", "Precision", "Recall", "F1-Score"]
).T

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df)

best_accuracy_model = results_df["Accuracy"].idxmax()
best_f1_model = results_df["F1-Score"].idxmax()

print("\nBest model by Accuracy:", best_accuracy_model)
print("Best model by F1-Score:", best_f1_model)

# Model performance visualization
results_df.plot(kind="bar", figsize=(10, 6))
plt.title("Classification Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()

# Confusion matrix plots
for name, cm in confusion_matrices.items():
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.yticks([0, 1], ["No", "Yes"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.show()

print("\nMACHINE LEARNING COMPLETED")

# ------------------------------------------------------------
# 7. K-MEANS CUSTOMER SEGMENTATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("7. K-MEANS CUSTOMER SEGMENTATION")
print("=" * 60)

cluster_features = df[
    ["age", "balance", "duration", "campaign"]
].copy()

scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_features)

# K = 4 selected for customer segmentation
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(cluster_scaled)

print("\nK-MEANS CLUSTER SIZES:")
print(df["Cluster"].value_counts().sort_index())

print("\nCLUSTER CHARACTERISTICS:")
cluster_summary = df.groupby("Cluster")[[
    "age",
    "balance",
    "duration",
    "campaign",
    "y"
]].mean()

print(cluster_summary)

# Cluster visualization
plt.figure(figsize=(8, 6))
plt.scatter(
    df["age"],
    df["balance"],
    c=df["Cluster"],
    alpha=0.5
)
plt.xlabel("Age")
plt.ylabel("Balance")
plt.title("K-Means Customer Segmentation")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

# Subscription rate by cluster
cluster_subscription = df.groupby("Cluster")["y"].mean() * 100

print("\nSUBSCRIPTION RATE BY CLUSTER (%):")
print(cluster_subscription)

print("\nK-MEANS CLUSTERING COMPLETED")

# ------------------------------------------------------------
# 8. FINAL OBSERVATIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("8. FINAL OBSERVATIONS")
print("=" * 60)

highest_cluster = cluster_subscription.idxmax()
lowest_cluster = cluster_subscription.idxmin()

print(
    f"\nHighest-potential cluster: Cluster {highest_cluster}"
    f" with {cluster_subscription[highest_cluster]:.2f}% subscription rate."
)

print(
    f"Lowest-response cluster: Cluster {lowest_cluster}"
    f" with {cluster_subscription[lowest_cluster]:.2f}% subscription rate."
)

print("\nRECOMMENDATION:")
print(
    "The bank should prioritize customers in the highest-potential "
    "cluster and use targeted marketing strategies. Customers in the "
    "lowest-response cluster should be approached with alternative "
    "strategies rather than repeated campaigns."
)

print("\n" + "=" * 60)
print("COMPLETE DATA SCIENCE WORKFLOW FINISHED SUCCESSFULLY")
print("=" * 60)
