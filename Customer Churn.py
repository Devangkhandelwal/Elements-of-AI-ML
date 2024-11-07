# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay  # Updated for confusion matrix visualization

# Step 1 & 2: Data Import and Basic Exploration
df = pd.read_csv('Bank Customer Churn Prediction.csv')  # Replace with actual file path
print(df.info())  # Check the data structure
print(df.describe())  # Basic statistics
print(df.isnull().sum())  # Check for missing values

# Step 3: Exploratory Data Analysis (EDA)
sns.countplot(x='churn', data=df)
plt.title("Churn Distribution")
plt.show()

# Visualizing distributions of numerical columns
df.hist(bins=20, figsize=(14,10))
plt.show()

# Step 4: Handling Missing Values and Outliers
# Assuming there are no missing values in this dataset
df.fillna(df.median(), inplace=True)
# Step 5: Feature Engineering and Encoding
# Convert categorical variables 'country' and 'gender' into numerical values using one-hot encoding
df = pd.get_dummies(df, columns=['country', 'gender'], drop_first=True)

