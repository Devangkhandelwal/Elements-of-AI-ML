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

# Step 5: Feature Engineering and Encoding
# Convert categorical variables 'country' and 'gender' into numerical values using one-hot encoding
df = pd.get_dummies(df, columns=['country', 'gender'], drop_first=True)

# Step 6: Feature Scaling
scaler = StandardScaler()
df[['credit_score', 'age', 'balance', 'estimated_salary']] = scaler.fit_transform(
    df[['credit_score', 'age', 'balance', 'estimated_salary']]
)

# Define X and y
X = df.drop(columns=['churn'])
y = df['churn']

# Step 7: Handling Imbalanced Data using SMOTE
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Step 8: Dimensionality Reduction Using PCA
pca = PCA(n_components=0.95)  # Adjust variance coverage as needed
X_res = pca.fit_transform(X_res)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Step 9: Model Building
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Step 10: Data Visualization and Model Evaluation
print(classification_report(y_test, predictions))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
ConfusionMatrixDisplay(conf_matrix).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# ROC Curve and AUC
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Step 11: Model Deployment
import joblib
joblib.dump(model, 'churn_prediction_model.pkl')

print("Model trained and evaluated successfully.")
