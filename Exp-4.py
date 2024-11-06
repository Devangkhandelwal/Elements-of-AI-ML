import pandas as pd

df = pd.read_csv('gym_members_exercise_tracking.csv')

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