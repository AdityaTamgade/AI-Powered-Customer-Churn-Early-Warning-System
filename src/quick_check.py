# src/quick_check.py
import pandas as pd
from sqlalchemy import create_engine
db_path = '../data/churn_db.sqlite'
engine = create_engine(f"sqlite:///{db_path}")

df = pd.read_sql("SELECT * FROM customers LIMIT 10", engine)
print(df.head())
print("\nDataset size:", pd.read_sql("SELECT COUNT(*) as cnt FROM customers", engine).iloc[0,0])
print("\nChurn distribution:")
print(pd.read_sql("SELECT churned, COUNT(*) as c FROM customers GROUP BY churned", engine))
