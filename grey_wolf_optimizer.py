import numpy as np
import pandas as pd
import random
import sys
import os
from Py_FS.wrapper.population_based.algorithm import Algorithm
from random_forest import FeatureSelection
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import hamming

class GreyWolfOptimizer(Algorithm):
    def __init__(
        self,
        num_agents,
        max_iter,
        train_data,
        train_label,
        test_data=None,
        test_label=None,
        seed=0,
        verbose=True,
        fs=None
    ):
        super().__init__(
            num_agents=num_agents,
            max_iter=max_iter,
            train_data=train_data,
            train_label=train_label,
            test_data=test_data,
            test_label=test_label,
            seed=seed,
            verbose=verbose
        )
        self.algo_name = 'GWO'
        self.agent_name = 'Wolf'
        # Use sigmoid transfer function for better exploration
        self.trans_function = lambda x: 1 / (1 + np.exp(-x))
        self.feature_names = list(train_data.columns)
        self.fs = fs
        self.history_global_best_fitness = []
        self.history_global_best_vector = []
        self.cur_iter = 0
        self.stagnation_count = 0  # Track stagnation for early stopping
        self.prev_best_fitness = -np.inf

    def initialize(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.num_features = self.train_data.shape[1]
        
        # Initialize population with wider feature range for better exploration
        self.population = initialize_population(
            self.num_agents,
            self.num_features,
            min_features=1,  # Allow single features
            max_features=int(0.8 * self.num_features)  # Up to 80% of features
        )
        
        # Initialize alpha/beta/delta positions and scores
        self.alpha_pos = np.zeros(self.num_features, dtype=int)
        self.alpha_score = -np.inf
        self.beta_pos = np.zeros(self.num_features, dtype=int)
        self.beta_score = -np.inf
        self.delta_pos = np.zeros(self.num_features, dtype=int)
        self.delta_score = -np.inf

    def obj_function(self, population):
        fitness = []
        # More flexible Random Forest parameters
        rf_params = dict(
            n_estimators=300,  # Increased trees
            max_depth=None,   # Allow deeper trees
            min_samples_split=5,
            criterion="gini",
            random_state=42
        )

        for mask in population:
            idx = np.where(mask == 1)[0]
            if idx.size == 0:
                fitness.append(0.0)
                continue

            # Use cross-validation for robust accuracy estimation
            Xtr = self.fs.x_train.iloc[:, idx].values
            ytr = self.fs.y_train.values
            rf = RandomForestClassifier(**rf_params)
            # 5-fold CV score
            acc = np.mean(cross_val_score(rf, Xtr, ytr, cv=5, scoring='accuracy'))

            # Reduced penalty to encourage larger feature sets
            penalty = 0.01 * (idx.size / self.num_features)
            # Diversity bonus: reward solutions different from alpha
            diversity = hamming(mask, self.alpha_pos) * 0.05 if self.cur_iter > 0 else 0
            fitness.append(acc - penalty + diversity)

        return np.array(fitness)

    def update_pack(self):
        # Evaluate fitness
        fitness = self.obj_function(self.population)
        avg_fit = np.mean(fitness)

        # Update alpha, beta, delta with diversity consideration
        sorted_indices = np.argsort(fitness)[::-1]
        for i in sorted_indices[:self.num_agents]:
            fit = fitness[i]
            if fit > self.alpha_score:
                self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos.copy()
                self.alpha_score, self.alpha_pos = fit, self.population[i].copy()
            elif fit > self.beta_score and hamming(self.population[i], self.alpha_pos) > 0.1:
                self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                self.beta_score, self.beta_pos = fit, self.population[i].copy()
            elif fit > self.delta_score and hamming(self.population[i], self.alpha_pos) > 0.1:
                self.delta_score, self.delta_pos = fit, self.population[i].copy()

        # Non-linear a for prolonged exploration
        a = 2 * (1 - (self.cur_iter / self.max_iter) ** 2)
        new_pop = []
        for wolf in self.population:
            updated = wolf.copy()
            for j in range(self.num_features):
                def move(leader):
                    r1, r2 = random.random(), random.random()
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = abs(C * leader[j] - wolf[j])
                    return leader[j] - A * D
                X1 = move(self.alpha_pos)
                X2 = move(self.beta_pos)
                X3 = move(self.delta_pos)
                val = (X1 + X2 + X3) / 3
                # Adaptive threshold for sigmoid
                threshold = self.trans_function(val) * (1 - self.cur_iter / self.max_iter)
                updated[j] = 1 if random.random() < threshold else 0
            
            # Mutation to maintain diversity
            if random.random() < 0.05:
                mutate_idx = random.randint(0, self.num_features - 1)
                updated[mutate_idx] = 1 - updated[mutate_idx]
            new_pop.append(updated)
        self.population = np.array(new_pop, dtype=int)

        # Update global best
        self.global_best = self.alpha_pos.copy()
        self.global_best_fitness = self.alpha_score
        self.history_global_best_fitness.append(self.global_best_fitness)
        self.history_global_best_vector.append(self.global_best.copy())

        # Early stopping check
        if abs(self.global_best_fitness - self.prev_best_fitness) < 1e-4:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        self.prev_best_fitness = self.global_best_fitness

        # Restart if stagnated
        if self.stagnation_count > 10:
            print("Stagnation detected, reinitializing population...")
            self.population = initialize_population(
                self.num_agents,
                self.num_features,
                min_features=1,
                max_features=int(0.8 * self.num_features)
            )
            self.stagnation_count = 0

        sel = [self.feature_names[i] for i, b in enumerate(self.global_best) if b]
        print(f"Iteration {self.cur_iter+1}: Best Fitness={self.global_best_fitness:.5f}, "
              f"Avg Fitness={avg_fit:.5f}, Selected Features ({len(sel)}): {', '.join(sel)}")
        self.cur_iter += 1

    def run(self):
        self.initialize()
        while self.cur_iter < self.max_iter:
            self.update_pack()
            if self.stagnation_count > 10:  # Additional safeguard
                break
        return self

# Helper function (updated)
def initialize_population(pop_size, num_features, min_features=1, max_features=None):
    if max_features is None:
        max_features = num_features
    pop = []
    for _ in range(pop_size):
        ind = np.zeros(num_features, dtype=int)
        k = random.randint(min_features, max_features)
        idx = random.sample(range(num_features), k)
        ind[idx] = 1
        pop.append(ind)
    # Ensure diversity by enforcing unique feature combinations
    pop = np.unique(np.array(pop, dtype=int), axis=0)
    while len(pop) < pop_size:
        ind = np.zeros(num_features, dtype=int)
        k = random.randint(min_features, max_features)
        idx = random.sample(range(num_features), k)
        ind[idx] = 1
        pop = np.vstack([pop, ind])
        pop = np.unique(pop, axis=0)
    return pop[:pop_size]

def plot_feature_importance(fs, mask, save_path=None):
    idx = [i for i,b in enumerate(mask) if b]
    if not idx:
        print("No features selected → skipping importance plot.")
        return
    X_sel = fs.x.iloc[:, idx]
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(X_sel, fs.y)
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    names = [fs.x.columns[i] for i in np.array(idx)[order]]

    plt.figure(figsize=(8,4))
    plt.bar(range(len(imp)), imp[order], align='center')
    plt.xticks(range(len(imp)), names, rotation=90)
    plt.title("Feature Importances")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def main():
    if os.path.exists("gwo_output.txt"):
        os.remove("gwo_output.txt")

    with open('gwo_output.txt', 'w') as f:
        orig = sys.stdout; sys.stdout = f
        try:
            fs = FeatureSelection(data_path='1000_encoded_gastric_cancer_data.csv')
            full = [1]*len(fs)
            base = round(fs.accuracy(full), 5)
            print(f"\nAccuracy using all {len(full)} features: {base}")

            gwo = GreyWolfOptimizer(num_agents=30, max_iter=50,
                                     train_data=fs.x, train_label=fs.y, fs=fs)
            gwo.run()

            best = gwo.global_best
            sel = [fs.x.columns[i] for i, b in enumerate(best) if b]
            fit = gwo.global_best_fitness
            print("\n------------- Leader Agent ------------------------")
            print(f"Fitness: {fit:.5f}")
            print(f"Vector: {best.tolist()}")
            print(f"Selected {len(sel)} features: {sel}")
            print("---------------------------------------------------")

            plot_feature_importance(fs, best, save_path="gwo_feature_importance.png")
            gwo.visualize(save_as="gwo_output.mov")

        finally:
            sys.stdout = orig

if __name__ == "__main__":
    main()
