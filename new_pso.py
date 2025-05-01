import numpy as np
import pandas as pd
from Py_FS.wrapper.population_based.algorithm import Algorithm
from Py_FS.wrapper.population_based._utilities import sort_agents
from Py_FS.wrapper.population_based._transfer_functions import get_trans_function
from new_rf import FeatureSelection


class ParticleSwarmOptimizer(Algorithm):
    def __init__(self,
                 num_agents,
                 max_iter,
                 train_data,
                 train_label,
                 test_data=None,
                 test_label=None,
                 save_conv_graph=False,
                 seed=0,
                 default_mode=False,
                 verbose=True):

        super().__init__(num_agents=num_agents,
                         max_iter=max_iter,
                         train_data=train_data,
                         train_label=train_label,
                         test_data=test_data,
                         test_label=test_label,
                         save_conv_graph=save_conv_graph,
                         seed=seed,
                         default_mode=default_mode,
                         verbose=verbose)

        self.algo_name = 'PSO'
        self.agent_name = 'Particle'
        self.trans_function = get_trans_function('s')  # use default without input

    def user_input(self):
        """Disabled user input — uses default transfer function."""
        pass

    def initialize(self):
        super().initialize()
        self.global_best_particle = [0] * self.num_features
        self.global_best_fitness = float("-inf")
        self.local_best_particle = [[0] * self.num_features for _ in range(self.num_agents)]
        self.local_best_fitness = [float("-inf")] * self.num_agents
        self.velocity = [[0.0] * self.num_features for _ in range(self.num_agents)]
        self.weight = 1.0

    def next(self):
        self.print('\n' + '=' * 80)
        self.print(f'Iteration - {self.cur_iter + 1}')
        self.print('=' * 80)

        self.weight = 1.0 - (self.cur_iter / self.max_iter)

        for i in range(self.num_agents):
            for j in range(self.num_features):
                self.velocity[i][j] *= self.weight
                r1, r2 = np.random.random(2)
                self.velocity[i][j] += r1 * (self.local_best_particle[i][j] - self.population[i][j])
                self.velocity[i][j] += r2 * (self.global_best_particle[j] - self.population[i][j])

        for i in range(self.num_agents):
            for j in range(self.num_features):
                trans_value = self.trans_function(self.velocity[i][j])
                self.population[i][j] = 1 if np.random.random() < trans_value else 0

        self.fitness = self.obj_function(self.population, self.training_data)
        self.population, self.fitness = sort_agents(self.population, self.fitness)

        for i in range(self.num_agents):
            if self.fitness[i] > self.local_best_fitness[i]:
                self.local_best_fitness[i] = self.fitness[i]
                self.local_best_particle[i] = self.population[i][:]
            if self.fitness[i] > self.global_best_fitness:
                self.global_best_fitness = self.fitness[i]
                self.global_best_particle = self.population[i][:]

        # Ensure binary format for output
        self.global_best_particle = [int(round(x)) for x in self.global_best_particle]

        self.cur_iter += 1


def main():
    # Step 1: Accuracy using all features
    fs = FeatureSelection(data_path='1000_encoded_gastric_cancer_data.csv')
    all_features_mask = [1] * len(fs)
    full_accuracy = round(fs.accuracy(all_features_mask), 5)
    print(f"\nAccuracy using all {len(fs.x.columns)} features: {full_accuracy}")

    # Step 2: Load data
    df = pd.read_csv('1000_encoded_gastric_cancer_data.csv')
    y = df.iloc[:, -1]
    x = df.iloc[:, :-1]
    feature_names = x.columns.tolist()

    # Step 3: Run PSO
    pso = ParticleSwarmOptimizer(num_agents=30, max_iter=20, train_data=x, train_label=y, default_mode=True)
    solution = pso.run()

    # Step 4: Interpret best solution
    raw_particle = np.array(solution.global_best_particle)
    selected_indices = [i for i, bit in enumerate(raw_particle) if bit == 1]
    selected_features = [feature_names[i] for i in selected_indices]

    print("\n------------- Leader Agent (from optimizer) ------------------------")
    print(f"Reported fitness (accuracy): {round(solution.global_best_fitness, 5)}")
    print(f"Feature vector: {raw_particle}")
    print(f"Reported feature count (non-zero entries): {np.count_nonzero(raw_particle)}")

    print("------------------------- Final PSO Output ---------------------------")
    print(f"Accuracy using PSO-selected features: {round(solution.global_best_fitness, 5)}")
    print(f"Number of selected features: {len(selected_features)}")
    print(f"Selected features: {selected_features}")
    print("---------------------------------------------------------------------")


if __name__ == '__main__':
    main()
