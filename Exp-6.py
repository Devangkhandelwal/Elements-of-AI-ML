# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('gym_members_exercise_tracking.csv')

# Display the basic dataset information
print("Number of rows and columns:")
print(df.shape)


print("\nFirst five rows of the dataset:")
print(df.head())

print("\nSize of the dataset (total number of elements):")
print(df.size)

print("\nNumber of missing values in each column:")
print(df.isnull().sum())

print("\nSum of numerical columns:")
print(df.select_dtypes(include='number').sum())

print("\nAverage of numerical columns:")
print(df.select_dtypes(include='number').mean())

print("\nMinimum values of numerical columns:")
print(df.select_dtypes(include='number').min())

print("\nMaximum values of numerical columns:")
print(df.select_dtypes(include='number').max())

# 1. Handling Missing Values

# Option 1: Drop rows with missing values
df_dropped = df.dropna()
print("\nDataset after dropping missing values:")
print(df_dropped)

# Option 2: Impute missing values with the mean (for numerical columns)
for column in df.select_dtypes(include='number').columns:
    df[column].fillna(df[column].mean(), inplace=True)

# Impute missing values for categorical columns with the most frequent value
for column in df.select_dtypes(include='object').columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

print("\nDataset after imputing missing values:")
print(df)

# 2. Handling Outliers

# Visualize the data to identify outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=df.select_dtypes(include='number'))
plt.title('Boxplot to Detect Outliers')
plt.show()

# Method 1: Removing outliers using the IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Apply outlier removal to numerical columns
for column in df.select_dtypes(include='number').columns:
    df = remove_outliers(df, column)

print("\nDataset after removing outliers:")
print(df)

# Method 2: Capping outliers (limiting values to the 5th and 95th percentiles)
def cap_outliers(df, column):
    lower_cap = df[column].quantile(0.05)
    upper_cap = df[column].quantile(0.95)
    df[column] = np.where(df[column] < lower_cap, lower_cap, df[column])
    df[column] = np.where(df[column] > upper_cap, upper_cap, df[column])
    return df

# Apply capping to numerical columns
for column in df.select_dtypes(include='number').columns:
    df = cap_outliers(df, column)

print("\nDataset after capping outliers:")
print(df)

# Final dataset after preprocessing
print("\nFinal preprocessed dataset:")
print(df)
