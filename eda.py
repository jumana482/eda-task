import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("student.csv")
print(df.head())

# Dataset information
print(df.info())

# Statistical summary
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Histogram
df.hist(figsize=(10,8))
plt.suptitle("Distribution of Features")
plt.show()

# Boxplot for Outlier Detection
plt.figure(figsize=(10,6))
sns.boxplot(data=df.select_dtypes(include='number'))
plt.title("Outlier Detection")
plt.show()