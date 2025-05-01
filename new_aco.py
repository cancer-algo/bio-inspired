import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ImprovedACOFeatureSelection:
    def __init__(self, data_path, n_ants=10, n_iterations=20, alpha=2.0, beta=1.0, evaporation_rate=0.3, q=0.5):
        self.data = pd.read_csv(data_path)
        self.y = self.data.iloc[:, -1]
        self.x = self.data.iloc[:, :-1]
        self.feature_names = self.x.columns.tolist()
        self.n_features = self.x.shape[1]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.pheromone = np.ones(self.n_features)
        self.heuristic = self._compute_heuristic_information()
        self.best_features = None
        self.best_score = 0.0

    def _compute_heuristic_information(self):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.x, self.y)
        importance = model.feature_importances_
        return importance / np.max(importance)

    def _initialize_ants(self):
        ants = []
        for _ in range(self.n_ants):
            feature_subset = np.zeros(self.n_features, dtype=int)
            for i in range(self.n_features):
                prob = (self.pheromone[i] ** self.alpha) * (self.heuristic[i] ** self.beta)
                prob = min(max(prob, 0.01), 0.99)  # Keep within bounds
                if np.random.rand() < prob:
                    feature_subset[i] = 1

            if not feature_subset.any():
                feature_subset[np.random.randint(0, self.n_features)] = 1

            # Force diversity if same as current best
            if self.best_features is not None and np.array_equal(feature_subset, self.best_features):
                idx_to_flip = np.random.choice(self.n_features, size=2, replace=False)
                feature_subset[idx_to_flip] = 1 - feature_subset[idx_to_flip]

            ants.append(feature_subset)
        return ants

    def _evaluate(self, feature_subset):
        selected_features = [i for i, bit in enumerate(feature_subset) if bit == 1]
        if not selected_features:
            return 0.0
        x_selected = self.x.iloc[:, selected_features]
        x_train, x_test, y_train, y_test = train_test_split(
            x_selected, self.y, test_size=0.3, random_state=np.random.randint(10000)
        )
        model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return accuracy_score(y_test, predictions)

    def _update_pheromone(self, ants, scores):
        self.pheromone *= (1 - self.evaporation_rate)
        for ant, score in zip(ants, scores):
            for i in range(self.n_features):
                if ant[i] == 1:
                    self.pheromone[i] += self.q * score

    def run(self):
        for iteration in range(self.n_iterations):
            ants = self._initialize_ants()
            scores = []
            for ant in ants:
                score = self._evaluate(ant)
                scores.append(score)
                if score > self.best_score:
                    self.best_score = score
                    self.best_features = ant.copy()
            self._update_pheromone(ants, scores)
            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best Score: {self.best_score:.4f}")
            print(f"Pheromone Sample: {np.round(self.pheromone[:5], 4)}")
        selected_feature_names = [name for i, name in enumerate(self.feature_names) if self.best_features[i] == 1]
        print(f"\nSelected Features ({len(selected_feature_names)}): {selected_feature_names}")
        print(f"Best Accuracy: {self.best_score:.4f}")

if __name__ == "__main__":
    aco_fs = ImprovedACOFeatureSelection(
        data_path='1000_encoded_gastric_cancer_data.csv',
        n_ants=20,
        n_iterations=30,
        alpha=2.0,
        beta=1.0,
        evaporation_rate=0.3,
        q=0.5
    )
    aco_fs.run()

