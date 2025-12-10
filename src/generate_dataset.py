import pandas as pd
import numpy as np
import os

# Ensure directories exist
os.makedirs("../data/raw", exist_ok=True)

# Total rows
n = 1000
np.random.seed(42)

customer_id = np.arange(1, n + 1)

gender = np.random.choice(["Male", "Female"], n)
age = np.random.randint(18, 70, n)
tenure = np.random.randint(1, 72, n)  # months
plan_type = np.random.choice(["Basic", "Standard", "Premium"], n, p=[0.4, 0.4, 0.2])

monthly_charges = np.round(np.random.uniform(200, 1500, n), 2)
total_charges = np.round(monthly_charges * tenure, 2)

num_complaints = np.random.poisson(1, n)
support_calls = np.random.poisson(2, n)

contract_type = np.random.choice(
    ["Monthly", "Quarterly", "Yearly"],
    n,
    p=[0.6, 0.25, 0.15]
)

payment_method = np.random.choice(
    ["Credit Card", "UPI", "Net Banking", "Wallet"],
    n
)

internet_usage_gb = np.round(np.random.uniform(5, 300, n), 1)

# Churn probability model
# Higher churn for:
# - Low tenure
# - High complaints
# - Basic plan
# - Monthly contract
base_prob = (
    (tenure < 12) * 0.25
    + (plan_type == "Basic") * 0.15
    + (contract_type == "Monthly") * 0.20
    + (num_complaints > 2) * 0.30
)

churned = np.random.binomial(1, np.clip(base_prob, 0, 0.8))

df = pd.DataFrame({
    "customer_id": customer_id,
    "gender": gender,
    "age": age,
    "tenure": tenure,
    "plan_type": plan_type,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "num_complaints": num_complaints,
    "support_calls": support_calls,
    "contract_type": contract_type,
    "payment_method": payment_method,
    "internet_usage_gb": internet_usage_gb,
    "churned": churned
})

output_path = "../data/raw/sample_customers.csv"
df.to_csv(output_path, index=False)

print(f"Dataset generated successfully → {output_path}")
print(df.head())
print("\nRows:", len(df))
