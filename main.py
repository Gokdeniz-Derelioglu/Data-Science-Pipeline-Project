from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch dataset 
iris = fetch_ucirepo(id=53)

# Data (features and targets as pandas DataFrames)
X = iris.data.features 
y = iris.data.targets 

# Print metadata
print("Metadata:")
print(iris.metadata)

# Print the first few rows of features and targets to check the data
print("\nFeatures (X):")
print(X.head())

print("\nTargets (y):")
print(y.head())

# Check the shape of the features (X) and targets (y)
print("Shape of Features (X):", X.shape)
print("Shape of Targets (y):", y.shape)

# Check for missing values
print(X.isnull().sum())
#Although there are no missing values, this code could have been used if there were any:
X.fillna(method='ffill', inplace=True)

iris_combined = pd.concat([X, y], axis=1)

# Histograms for each feature
X.hist(figsize=(10, 8))
plt.show()

# Pairplot to visualize relationships between features and class
sns.pairplot(iris_combined, hue='class')
plt.show()

# Boxplot to see distribution across the features
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_combined)
plt.xticks(rotation=45)
plt.show()

# Correlation matrix with heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.show()