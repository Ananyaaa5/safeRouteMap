import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Load the Iris dataset
iris = load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['species'] = iris.target_names[iris.target]

print("Dataset loaded successfully.")

# 2. Perform basic data exploration
print("\n--- Basic Data Exploration ---")
print("\nMissing values:")
print(df_iris.isnull().sum())

print("\nData types:")
print(df_iris.dtypes)

print("\nSummary statistics:")
print(df_iris.describe())

# 3. Create visualizations
print("\n--- Visualizations ---")

# Histograms for numerical features
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df_iris[feature], kde=True)
    plt.title(f'Histogram of {feature}')
plt.tight_layout()
plt.show()

# Scatter plot for relationships between features (e.g., sepal length vs. sepal width)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=df_iris)
plt.title('Scatter Plot of Sepal Length vs. Sepal Width')
plt.show()

# Box plots to understand distribution across species
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=feature, data=df_iris)
    plt.title(f'Box Plot of {feature} by Species')
plt.tight_layout()
plt.show()