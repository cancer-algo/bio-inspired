# https://ieeexplore.ieee.org/abstract/document/6620037

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif

# Load and preprocess the dataset
data = pd.read_csv('breast_cancer_data.csv')
data['diagnosis'] = LabelEncoder().fit_transform(data['diagnosis'])

if 'id' in data.columns:
    data.drop(columns=['id'], inplace=True)

x = data.drop(columns=['diagnosis'])
y = data['diagnosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

n_features = x.shape[1]
feature_names = x.columns

# ACO parameters
n_ants = 20
n_iterations = 30
alpha = 1.0
beta = 2.0
rho = 0.1
tau0 = 0.01
subset_size = 10

# Initialize pheromone and heuristic (visibility)
pheromone = np.full(n_features, tau0)
visibility = mutual_info_classif(x_train, y_train)
visibility = visibility / (np.max(visibility) + 1e-9)

# Evaluation function
def evaluate(feature_indices):
    if len(feature_indices) == 0:
        return 0
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
    clf.fit(x_train.iloc[:, feature_indices], y_train)
    preds = clf.predict(x_test.iloc[:, feature_indices])
    return accuracy_score(y_test, preds)

# ACO loop
rng = np.random.default_rng(seed=42)  # Create a random number generator with a seed
best_features = []
best_score = 0

for iteration in range(n_iterations):
    all_solutions = []
    all_scores = []

    for ant in range(n_ants):
        selected = []
        unvisited = list(range(n_features))

        while len(selected) < subset_size and unvisited:
            probs = []
            for j in unvisited:
                tau = pheromone[j] ** alpha
                eta = visibility[j] ** beta
                probs.append(tau * eta)
            probs = np.array(probs)
            probs /= probs.sum()
            chosen_idx = rng.choice(len(unvisited), p=probs)
            chosen_idx = rng.choice(len(unvisited), p=probs)
            selected_feature = unvisited.pop(chosen_idx)
            selected.append(selected_feature)

        score = evaluate(selected)
        all_solutions.append(selected)
        all_scores.append(score)

        if score > best_score:
            best_score = score
            best_features = selected.copy()

    pheromone *= (1 - rho)
    for features, score in zip(all_solutions, all_scores):
        for f in features:
            pheromone[f] += score

    print(f"Iteration {iteration + 1}/{n_iterations}, Best Score: {best_score:.5f}")

best_features = sorted(best_features)
selected_features = np.array(best_features)

# Re-evaluate the same best feature subset (to mirror PSO logic)
clf_final = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
clf_final.fit(x_train.iloc[:, selected_features], y_train)
y_pred = clf_final.predict(x_test.iloc[:, selected_features])
final_accuracy = accuracy_score(y_test, y_pred)

# Final output
print(f"\nSelected features indices: {selected_features}")
# print(f"Feature names: {list(x.columns[selected_features])}")
print(f"Number of selected features: {len(selected_features)}")
print(f"Accuracy with selected features: {final_accuracy:.5f}")
print(f"Best accuracy score during ACO search (training): {best_score:.5f}")


"""
Runs for 30 iterations; fitting on training data; predicting on test data
And using the same features from best_features
Gives output - Selected features indices: [ 0  2  3  6  7 13 20 22 23 26]
Number of selected features: 10
Accuracy with selected features: 0.95322
Best accuracy score during ACO search (training): 0.97076
"""



