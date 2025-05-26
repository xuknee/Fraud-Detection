import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
data = pd.read_csv('creditcard.csv')

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