# src/generate_sample_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

os.makedirs('../data/raw', exist_ok=True)

n = 1000
np.random.seed(42)

customer_id = np.arange(1, n+1)
signup_date = [ (datetime.today() - timedelta(days=int(np.random.exponential(365)))) .date() for _ in range(n)]
birth_year = np.random.randint(1960, 2005, size=n)
gender = np.random.choice(['Male','Female','Other'], size=n, p=[0.48,0.48,0.04])
plan_type = np.random.choice(['basic','standard','premium'], size=n, p=[0.4,0.45,0.15])
monthly_charges = np.round(np.random.normal(40,15,size=n),2)
monthly_charges = np.clip(monthly_charges, 5, 200)
last_payment_delay = np.random.poisson(0.5, size=n)
avg_usage = np.round(np.random.exponential(50, size=n),2)
support_tickets_30d = np.random.poisson(0.2, size=n)
# churn label (synthetic): higher chance if delay and low usage
churn_prob = 0.1 + (last_payment_delay * 0.15) + (support_tickets_30d * 0.05) + (np.where(avg_usage<10,0.2,0))
churned = np.random.rand(n) < churn_prob

df = pd.DataFrame({
    'customer_id': customer_id,
    'signup_date': signup_date,
    'birth_year': birth_year,
    'gender': gender,
    'plan_type': plan_type,
    'monthly_charges': monthly_charges,
    'last_payment_delay': last_payment_delay,
    'avg_usage': avg_usage,
    'support_tickets_30d': support_tickets_30d,
    'churned': churned.astype(int)
})

df.to_csv('../data/raw/sample_customers.csv', index=False)
print("Sample CSV created at data/raw/sample_customers.csv")
