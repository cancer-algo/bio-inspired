import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
data = pd.read_csv('breast_cancer_data.csv')
data['diagnosis'] = LabelEncoder().fit_transform(data['diagnosis'])

# Drop the 'id' column if present
if 'id' in data.columns:
    data.drop(columns=['id'], inplace=True)

# Separate features and target
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

# Train Random Forest classifier using all features
clf = RandomForestClassifier(
    n_estimators=100, max_depth=2, criterion='gini', random_state=42
)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy using all features: {accuracy:.5f}")

# Gives output - Accuracy using all 30 features: 0.95322