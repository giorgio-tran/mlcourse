import numpy as np
import pandas as pd
import seaborn as sns

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

# Contingency table, comparing two variables
print(df["International plan"])
print(pd.crosstab(df["Churn"], df["International plan"]))
# normalize to see proportions
print(pd.crosstab(df["Churn"], df["International plan"], normalize=True))

# Pivot table
# Row, column, then specify statistic to calculate
print(df.pivot_table(["Total day calls", "Total eve calls", "Total night calls"], ["Area code"], aggfunc="mean"))

# DataFrame transformations
total_calls = (
  df["Total day calls"]
  + df["Total eve calls"]
  + df["Total night calls"]
  + df["Total intl calls"]
)
# loc specifies which location to insert the new column
df.insert(loc=len(df.columns), column="Total calls", value=total_calls)
print(df.head())

df["Total charge"] = (
    df["Total day charge"]
    + df["Total eve charge"]
    + df["Total night charge"]
    + df["Total intl charge"]
)
df.head()

# removing/dropping columns. Need to specify axis=1 to drop columns, otherwise it will drop rows
print(df.drop(["Total charge", "Total calls"], axis=1, inplace=True))
df.head()

# predicting telecom churn
pd.crosstab(df["Churn"], df["International plan"], margins=True)

sns.set_theme()
# Graphis in retina format
%config InlineBacked.figure_format = 'retina'

sns.countplot(x="International plan", hue="Churn", data=df)

# check customer service
pd.crosstab(df["Churn"], df["Customer service calls"], margins=True)
sns.countplot(x="Customer service calls", hue="Churn", data=df)

# Add many service calls column that looks at data of customer service calls and checks if it is greater than 3
df["Many_service_calls"] = (df["Customer service calls"] > 3).astype("int")
pd.crosstab(df["Many_service_calls"], df["Churn"], margins=True)
sns.countplot(x="Many_service_calls", hue="Churn", data=df)

pd.crosstab(df["Many_service_calls"] & df["International plan"], df["Churn"])



