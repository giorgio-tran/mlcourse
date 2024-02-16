import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"

df = pd.read_csv(DATA_URL + "telecom_churn.csv")
print(df.head())

# Get shape, (rows, columns)
print(df.shape)

# Get information
print(df.info())

# Convert data types
df["Churn"] = df["Churn"].astype("int64")
df.describe()

# describe data such as astype, mean, std, min, max, 25%, 50%, 75%
print(df.describe())

# to see non-numerical stats, need to explicitly include
print(df.describe(include=["object", "bool"]))

# counts number of customers who churned (1) or not (0)
print(df["Churn"].value_counts())

# Fraction
print(df["Churn"].value_counts(normalize=True))

# Sorting
print(df.sort_values(by="Total day charge", ascending=False).head())

# Filtering
print(df.select_dtypes(include=np.number)[df["Churn"] == 1].mean())

print(df[(df["Churn"] == 0) & (df["International plan"] == "No")]["Total intl minutes"].max())

#Indexing (column, row)
print(df.loc[0:5, "State": "Area code"])

# apply method to apply functions to each column
# lambda is an anonymous function
print(df[df["State"].apply(lambda state: state[0] == "W")].head())

# map in python
d = {"No": False, "Yes": True}
df["International plan"] = df["International plan"].map(d)
print(df.head())
# using replace
df = df.replace({"Voice mail plan": d})

# difference between map and replace is that map replace null values with NaN, and replace just ignores them
a_series = pd.Series(['a', 'b', 'c'])
a_series.replace({'a': 1, 'b': 1})     # 1, 2, c
a_series.map({'a': 1, 'b': 2})     # 1, 2, NaN

# grouping
columns_to_show = ["Total day minutes", "Total eve minutes", "Total night minutes"]

print(df.groupby(["Churn"])[columns_to_show].describe(percentiles=[]))

# agg, allows you to add functions to specify what exactly you want to see. In this case we want to see the mean, std, min, and max
print(df.groupby(["Churn"])[columns_to_show].agg([np.mean, np.std, np.min, np.max]))