# src/data_ingest.py
import pandas as pd
from sqlalchemy import create_engine
import os

csv_path = os.path.join('..','data','raw','sample_customers.csv')
df = pd.read_csv(csv_path)

# create a local SQLite database file
db_path = os.path.join('..','data','churn_db.sqlite')
engine = create_engine(f"sqlite:///{db_path}")

# write to SQL table
df.to_sql('customers', con=engine, if_exists='replace', index=False)
print("Loaded to SQLite DB at", db_path)

# quick query to show counts
q = "SELECT COUNT(*) as cnt FROM customers"
print(pd.read_sql(q, engine))
