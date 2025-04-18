# https://pyswarms.readthedocs.io/en/development/examples/feature_subset_selection.html
# https://joss.theoj.org/papers/10.21105/joss.00433

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from pyswarms.discrete import BinaryPSO

# Load and preprocess the dataset
data = pd.read_csv('breast_cancer_data.csv')
data['diagnosis'] = LabelEncoder().fit_transform(data['diagnosis'])

# Drop the 'id' column
if 'id' in data.columns:
    data.drop(columns=['id'], inplace=True)

# Separate features and target
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

# Define the objective function for PSO
def objective_function(particles):
    n_particles = particles.shape[0]
    scores = []

    for i in range(n_particles):
        # Select features where the particle has a value of 1
        mask = particles[i].astype(bool)
        if np.count_nonzero(mask) == 0:
            scores.append(0)
            continue

        # Train the classifier with selected features
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=2, criterion='gini', random_state=42
        )
        clf.fit(X_train.iloc[:, mask], y_train)
        y_pred = clf.predict(X_test.iloc[:, mask])
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)

    # Since PySwarms minimizes the objective, return negative accuracy
    return -np.array(scores)

# Initialize Binary PSO
n_features = X.shape[1]
options = {'c1': 2, 'c2': 2, 'w': 0.9, 'k': 5, 'p': 2}
optimizer = BinaryPSO(n_particles=30, dimensions=n_features, options=options)

# Perform optimization
cost, pos = optimizer.optimize(objective_function, iters=20)

# Evaluate the selected features
selected_features = np.nonzero(pos == 1)[0]
print(f"Selected features indices: {selected_features}")
print(f"Number of selected features: {len(selected_features)}")

# Train and evaluate the classifier with selected features
clf = RandomForestClassifier(
    n_estimators=100, max_depth=2, criterion='gini', random_state=42
)
clf.fit(X_train.iloc[:, selected_features], y_train)
y_pred = clf.predict(X_test.iloc[:, selected_features])
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with selected features: {final_accuracy:.5f}")

""" Gives output - Selected features indices: [ 0  1  3  4  5  8 11 12 14 15 16 19 20 21 22 23 27 28 29]
Number of selected features: 19
Accuracy with selected features: 0.98246 """
