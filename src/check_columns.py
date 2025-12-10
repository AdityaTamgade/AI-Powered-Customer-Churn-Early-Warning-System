import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("sqlite:///data/churn_db.sqlite")
df = pd.read_sql("SELECT * FROM customers LIMIT 5", engine)

print("\nColumns in SQL table:")
print(df.columns.tolist())

print("\nSample rows:")
print(df.head())
