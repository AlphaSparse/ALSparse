import pandas as pd
import numpy as np

df1 = pd.read_csv("merge_no_share.csv")
df2 = pd.read_csv("merge_static_share.csv")

df = pd.merge(df1, df2, on="mtx_path")

df["speedup"] = df["alphasparse_no_share"]/df["alphasparse_static_share"]

print(df)
print(df["speedup"].mean())
