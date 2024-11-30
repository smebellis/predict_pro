import pandas as pd
import os
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack, csr_matrix
import shap
import matplotlib.pyplot as plt

PICKLE_DIR = "pickle_files"


# Function to load pickle files
def load_as_pkl(file_name, pickle_dir):
    with open(os.path.join(pickle_dir, f"{file_name}.pkl"), "rb") as f:
        return pickle.load(f)


# Load the dataframes
train_df = load_as_pkl("train", PICKLE_DIR)
val_df = load_as_pkl("val", PICKLE_DIR)
test_df = load_as_pkl("test", PICKLE_DIR)

# Drop the 'POLYLINE' column since it can't be directly encoded
train_df = train_df.drop(columns=["POLYLINE"])
val_df = val_df.drop(columns=["POLYLINE"])
test_df = test_df.drop(columns=["POLYLINE"])

# Separate features (X) and target (y)
y_train = train_df["TRAFFIC_STATUS"]
y_val = val_df["TRAFFIC_STATUS"]
y_test = test_df["TRAFFIC_STATUS"]

X_train = train_df.drop(columns=["TRAFFIC_STATUS"])
X_val = val_df.drop(columns=["TRAFFIC_STATUS"])
X_test = test_df.drop(columns=["TRAFFIC_STATUS"])

# Identify categorical features that need encoding
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns

# Ensure that all numeric columns are of numeric dtype
numeric_cols = X_train.select_dtypes(include=["number"]).columns

# Convert numeric columns to a proper dtype and handle errors
for col in numeric_cols:
    if X_train[col].dtype == "O":
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_val[col] = pd.to_numeric(X_val[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

# Fill any NaN values in the numeric data with 0 (or any other placeholder value)
X_train[numeric_cols] = X_train[numeric_cols].fillna(0)
X_val[numeric_cols] = X_val[numeric_cols].fillna(0)
X_test[numeric_cols] = X_test[numeric_cols].fillna(0)

# Apply OneHotEncoder to categorical columns, ensuring consistency across splits
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

# Fit the encoder on the training data only
X_train_encoded = encoder.fit_transform(X_train[categorical_cols])

# Transform validation and test data using the same encoder
X_val_encoded = encoder.transform(X_val[categorical_cols])
X_test_encoded = encoder.transform(X_test[categorical_cols])

# Drop the original categorical columns from numeric data
X_train_numeric = X_train.drop(columns=categorical_cols)
X_val_numeric = X_val.drop(columns=categorical_cols)
X_test_numeric = X_test.drop(columns=categorical_cols)

# Convert all numeric data to float64 to avoid data type issues
X_train_numeric = X_train_numeric.astype(np.float64)
X_val_numeric = X_val_numeric.astype(np.float64)
X_test_numeric = X_test_numeric.astype(np.float64)

# Convert numeric data to sparse format for efficient concatenation with encoded features
X_train_numeric_sparse = csr_matrix(X_train_numeric.values)
X_val_numeric_sparse = csr_matrix(X_val_numeric.values)
X_test_numeric_sparse = csr_matrix(X_test_numeric.values)

# Concatenate numeric and encoded categorical features using sparse matrix
X_train_final = hstack([X_train_encoded, X_train_numeric_sparse])
X_val_final = hstack([X_val_encoded, X_val_numeric_sparse])
X_test_final = hstack([X_test_encoded, X_test_numeric_sparse])

# Train the model
model = RandomForestClassifier(random_state=42, n_jobs=-1)
model.fit(X_train_final, y_train)

# Evaluate the model
train_score = model.score(X_train_final, y_train)
val_score = model.score(X_val_final, y_val)

print(f"Training Accuracy: {train_score:.2f}")
print(f"Validation Accuracy: {val_score:.2f}")

# Add SHAP analysis
# Create a SHAP explainer for the trained RandomForest model
explainer = shap.TreeExplainer(model)

# Take a sample of the validation set to calculate SHAP values
X_val_sample = X_val_final[:100, :].toarray()  # Convert to dense for SHAP compatibility

# Calculate SHAP values for the sample
shap_values = explainer.shap_values(X_val_sample)

# Generate SHAP summary plot to visualize feature importance
plt.title("SHAP Feature Importance Summary (Validation Sample)")
shap.summary_plot(
    shap_values,
    X_val_sample,
    feature_names=encoder.get_feature_names_out(categorical_cols),
)
