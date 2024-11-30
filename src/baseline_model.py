import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# from helper import plot_metrics
import matplotlib.pyplot as plt

# Initialize the metrics dictionary
metrics = {}

# Load train and test data from pickle files
train_df = pd.read_pickle("pickle_files/train.pkl")
test_df = pd.read_pickle("pickle_files/test.pkl")

# Extract features and labels
X_train = train_df.drop(columns=["TRAFFIC_STATUS"])
y_train = train_df["TRAFFIC_STATUS"]

X_test = test_df.drop(columns=["TRAFFIC_STATUS"])
y_test = test_df["TRAFFIC_STATUS"]


# Zero-Rule Classifier (Most Frequent)
zero_rule_clf = DummyClassifier(strategy="most_frequent")
zero_rule_clf.fit(X_train, y_train)
y_pred_zero_rule = zero_rule_clf.predict(X_test)

# Calculate metrics for Zero-Rule Classifier
metrics["Zero_Rule"] = {
    "Accuracy": accuracy_score(y_test, y_pred_zero_rule),
    "Precision": precision_score(
        y_test, y_pred_zero_rule, average="weighted", zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred_zero_rule, average="weighted", zero_division=0
    ),
    "F1 Score": f1_score(y_test, y_pred_zero_rule, average="weighted"),
    "Confusion Matrix": confusion_matrix(y_test, y_pred_zero_rule),
}

# Random Weight Classifier
random_clf = DummyClassifier(strategy="uniform")
random_clf.fit(X_train, y_train)
y_pred_random = random_clf.predict(X_test)

# Calculate metrics for Random Classifier
metrics["Random_Classifier"] = {
    "Accuracy": accuracy_score(y_test, y_pred_random),
    "Precision": precision_score(
        y_test, y_pred_random, average="weighted", zero_division=0
    ),
    "Recall": recall_score(y_test, y_pred_random, average="weighted", zero_division=0),
    "F1 Score": f1_score(y_test, y_pred_random, average="weighted"),
    "Confusion Matrix": confusion_matrix(y_test, y_pred_random),
}


# Display the metrics
for clf_name, clf_metrics in metrics.items():
    print(f"\nPerformance Metrics for {clf_name}:\n")
    for metric, value in clf_metrics.items():
        if metric == "Confusion Matrix":
            print(f"{metric}:\n{value}\n")
        else:
            print(f"{metric}: {value}")

# Save only scalar metrics to a DataFrame for easier visualization
metrics_df = pd.DataFrame(
    {
        k: {metric: val for metric, val in v.items() if metric != "Confusion Matrix"}
        for k, v in metrics.items()
    }
)
print(metrics_df)

# # Save metrics to a pickle file
# metrics_df.to_pickle("pickle_files/baseline_metrics.pkl")

# plot_metrics(metrics)
