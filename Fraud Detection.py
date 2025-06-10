import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score

# Load Data
data = pd.read_csv('/Users/johnny/Downloads/creditcard.csv')

# Display basic info
print(f"Total transactions: {len(data)}")
print(f"Number of features: {data.shape[1]}")
print("\nFirst 5 rows:")
print(data.head())

# Calculate fraud percentage
fraud_count = data['Class'].sum()
total_count = len(data)
fraud_percentage = (fraud_count / total_count) * 100

print("\nFraud Analysis:")
print("==============")
print(f"Genuine transactions: {total_count - fraud_count} ({100 - fraud_percentage:.2f}%)")
print(f"Fraudulent transactions: {fraud_count} ({fraud_percentage:.2f}%)")

# Set style
sns.set_style("whitegrid")

# Create figure
plt.figure(figsize=(12, 5))

# Transaction amount distribution
plt.subplot(1, 2, 1)
sns.histplot(data=data[data['Class'] == 0]['Amount'], bins=50, color='green', label='Genuine')
sns.histplot(data=data[data['Class'] == 1]['Amount'], bins=50, color='red', label='Fraud')
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount ($)')
plt.ylabel('Count')
plt.legend()
plt.yscale('log')  # Using log scale for better visualization

# Time distribution
plt.subplot(1, 2, 2)
sns.histplot(data=data[data['Class'] == 0]['Time'], bins=50, color='green', label='Genuine')
sns.histplot(data=data[data['Class'] == 1]['Time'], bins=50, color='red', label='Fraud')
plt.title('Transaction Time Distribution')
plt.xlabel('Time (seconds since first transaction)')
plt.ylabel('Count')
plt.legend()

plt.tight_layout()
plt.show()

# Create amount ranges
bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
labels = ['0-10', '10-50', '50-100', '100-500', '500-1000', 
          '1000-5000', '5000-10000', '10000-50000', '50000+']

data['AmountRange'] = pd.cut(data['Amount'], bins=bins, labels=labels)

# Calculate fraud percentage by amount range
fraud_by_amount = data.groupby('AmountRange')['Class'].agg(['sum', 'count'])
fraud_by_amount['percentage'] = (fraud_by_amount['sum'] / fraud_by_amount['count']) * 100

print("\nFraud Percentage by Transaction Amount:")
print("======================================")
print(fraud_by_amount[['sum', 'percentage']].sort_values('percentage', ascending=False))

# Preprocessing with sklearn
# Scale 'Amount' and 'Time' features
scaler = RobustScaler()
data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

# Drop original columns
data.drop(['Time', 'Amount'], axis=1, inplace=True)

# Split data into features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nData Splitting:")
print("==============")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Fraud cases in test set: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)")

# Comparing with a Dummy Classifier 
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='stratified')
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)

print("\nDummy Classifier Performance:")
print("============================")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Genuine', 'Fraud'], 
            yticklabels=['Genuine', 'Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Plot precision-recall curve
y_scores = dummy.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
average_precision = average_precision_score(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP={average_precision:.2f})')
plt.show()
