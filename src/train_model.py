import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------
# 1️⃣ Load processed data
# ----------------------------------
df = pd.read_csv("data/processed/processed_customers.csv")

# Target column
y = df["churned"]

# Features
X = df.drop(columns=["churned"])

# Save feature names before scaling (used by Streamlit app)
feature_names = list(X.columns)

with open("models/feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=4)

# ----------------------------------
# 2️⃣ Train-test split
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# 3️⃣ Scale features
# ----------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------
# 4️⃣ Train Random Forest on SCALED DATA
# ----------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

rf_model.fit(X_train_scaled, y_train)

# ----------------------------------
# 5️⃣ Save trained objects
# ----------------------------------
joblib.dump(rf_model, "models/model.pkl")       # main model for Streamlit app
joblib.dump(scaler, "models/scaler.pkl")        # scaler for app

print("🎉 Training Completed Successfully!")
print("👉 model.pkl, scaler.pkl, feature_names.json saved in /models/")
