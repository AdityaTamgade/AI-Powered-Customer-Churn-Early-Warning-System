import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(file_path):
    """Load raw churn dataset."""
    print("📂 Loading data from:", file_path)
    return pd.read_csv(file_path)

def clean_data(df):
    """Handle missing values and correct data types."""

    # Fill numeric missing values
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill categorical missing values
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def feature_engineering(df):
    """Add useful churn features."""

    # Flag high usage users
    df["high_usage_flag"] = (df["internet_usage_gb"] > df["internet_usage_gb"].median()).astype(int)

    # Flag users who complain a lot
    df["high_complaints_flag"] = (df["num_complaints"] >= 3).astype(int)

    # Flag heavy support callers
    df["high_support_flag"] = (df["support_calls"] >= 5).astype(int)

    # Tenure bucket
    df["tenure_months"] = df["tenure"] * 12

    return df

def encode_categories(df):
    """Label encode categorical columns."""
    encoders = {}
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders

def scale_data(df, target="churned"):
    """Scale numeric features."""
    scaler = StandardScaler()

    X = df.drop(columns=[target])
    y = df[target]

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def split_data(X, y):
    """Train-test split."""
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_pipeline(file_path):
    """Full pipeline."""
    df = load_data(file_path)
    df = clean_data(df)
    df = feature_engineering(df)
    df, encoders = encode_categories(df)
    X, y, scaler = scale_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save processed dataset
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/processed_customers.csv", index=False)
    print("✅ Processed file saved at: data/processed/processed_customers.csv")

    return X_train, X_test, y_train, y_test, scaler, encoders, df


if __name__ == "__main__":
    file_path = "data/raw/sample_customers.csv"
    preprocess_pipeline(file_path)
