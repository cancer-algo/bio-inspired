import numpy as np
import pandas as pd
import sys
from Py_FS.wrapper.population_based.algorithm import Algorithm
from Py_FS.wrapper.population_based._utilities import sort_agents
from random_forest import FeatureSelection


class AntColonyOptimizer(Algorithm):
    def __init__(
        self,
        num_agents,
        max_iter,
        train_data,
        train_label,
        test_data=None,
        test_label=None,
        save_conv_graph=False,
        seed=0,
        default_mode=False,
        verbose=True
    ):
        super().__init__(
            num_agents=num_agents,
            max_iter=max_iter,
            train_data=train_data,
            train_label=train_label,
            test_data=test_data,
            test_label=test_label,
            save_conv_graph=save_conv_graph,
            seed=seed,
            default_mode=default_mode,
            verbose=verbose
        )
        self.algo_name = 'ACO'
        self.agent_name = 'Ant'
        self.evaporation_rate = 0.1
        self.alpha = 1.0
        self.beta = 2.0

    def initialize(self):
        super().initialize()
        self.pheromone = [1.0] * self.num_features
        self.heuristic = [1.0] * self.num_features
        self.global_best = [0] * self.num_features
        self.global_best_fitness = float('-inf')
        self.weight = None

    def update_colony(self):
        header = f"\n{'='*80}\nIteration - {self.cur_iter + 1}\n{'='*80}"
        self.print(header)

        pher = np.array(self.pheromone)
        heur = np.array(self.heuristic)
        probs = (pher ** self.alpha) * (heur ** self.beta)
        probs = probs / np.sum(probs)

        new_pop = []
        for _ in range(self.num_agents):
            particle = [1 if np.random.random() < p else 0 for p in probs]
            new_pop.append(particle)
        self.population = np.array(new_pop)

        self.fitness = self.obj_function(self.population, self.training_data)
        self.population, self.fitness = sort_agents(self.population, self.fitness)

        best_fit = self.fitness[0]
        best_part = self.population[0][:]
        if best_fit > self.global_best_fitness:
            self.global_best_fitness = best_fit
            self.global_best = best_part[:]

        self.pheromone = [(1 - self.evaporation_rate) * t for t in self.pheromone]
        for idx, bit in enumerate(self.global_best):
            if bit:
                self.pheromone[idx] += self.global_best_fitness

        self.cur_iter += 1

    next = update_colony


def main():
    with open('aco_output.txt', 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            fs = FeatureSelection(data_path='1000_encoded_gastric_cancer_data.csv')
            full_mask = [1] * len(fs)
            baseline = round(fs.accuracy(full_mask), 5)
            print(f"\nAccuracy using all {len(fs.x.columns)} features: {baseline}")

            df = pd.read_csv('1000_encoded_gastric_cancer_data.csv')
            X, y = df.iloc[:, :-1], df.iloc[:, -1]
            feature_names = X.columns.tolist()

            aco = AntColonyOptimizer(
                num_agents=30,
                max_iter=20,
                train_data=X,
                train_label=y,
                default_mode=True
            )
            solution = aco.run()

            best_vec = np.array(solution.global_best)
            selected = [feature_names[i] for i, bit in enumerate(best_vec) if bit]
            best_acc = round(solution.global_best_fitness, 5)

            print("\n------------- Leader Agent ------------------------")
            print(f"Fitness: {best_acc}")
            print(f"Vector: {best_vec}")
            print(f"Selected {len(selected)} features: {selected}")
            print("---------------------------------------------------")
        finally:
            sys.stdout = original_stdout


if __name__ == '__main__':
    main()
