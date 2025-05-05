import numpy as np
import pandas as pd
import sys
from Py_FS.wrapper.population_based.algorithm import Algorithm
from Py_FS.wrapper.population_based._utilities import sort_agents
from random_forest import FeatureSelection
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers


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
        evaporation_rate=0.4,
        max_pheromone=5,
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

        # ACO-specific parameters
        self.evaporation_rate = 9.0
        self.alpha = 1.0
        self.beta = 1.0
        self.min_pheromone = 0.01
        self.max_pheromone = 5.0
        self.lambda_penalty = 0.01
        self.reset_interval = 5

        if hasattr(train_data, 'columns'):
            self.feature_names = list(train_data.columns)
        else:
            self.feature_names = [str(i) for i in range(self.num_features)]

        # History buffers for visualization
        self.history_selected_features = []
        self.history_global_best_fitness = []
        self.history_pheromone = []

    def initialize(self):
        super().initialize()

        # initialize pheromone and heuristic information
        self.pheromone = [1.0] * self.num_features
        from sklearn.feature_selection import mutual_info_classif
        self.heuristic = mutual_info_classif(self.train_data, self.train_label).tolist()

        self.global_best = [0] * self.num_features
        self.global_best_fitness = float('-inf')

    def obj_function(self, population, training_data):
        fitness = []
        for solution in population:
            acc = self.wrapper_function(solution)
            size_penalty = self.lambda_penalty * (np.sum(solution) / self.num_features)
            fitness.append(acc - size_penalty)
        return np.array(fitness)

    def update_colony(self):

        # periodically reset pheromones to avoid premature convergence
        if self.cur_iter > 0 and self.cur_iter % self.reset_interval == 0:
            self.pheromone = [1.0] * self.num_features

        # compute probabilities based on pheromone and heuristic
        pher = np.clip(np.array(self.pheromone), self.min_pheromone, self.max_pheromone)
        heur = np.array(self.heuristic) + 1e-10  # avoid division by zero
        probs = (pher ** self.alpha) * (heur ** self.beta)
        probs = probs / np.sum(probs)

        # build new population
        new_pop = []
        for _ in range(self.num_agents):
            ant = [1 if np.random.random() < p else 0 for p in probs]
            if sum(ant) == 0:
                ant[np.random.randint(self.num_features)] = 1  # ensure at least one feature
            new_pop.append(ant)
        self.population = np.array(new_pop)

        # evaluate fitness and sort
        self.fitness = self.obj_function(self.population, self.training_data)
        avg_fitness = np.mean(self.fitness) 
        self.population, self.fitness = sort_agents(self.population, self.fitness)

        # update global best
        best_fit = self.fitness[0]
        best_ant = self.population[0][:]
        if best_fit > self.global_best_fitness:
            self.global_best_fitness = best_fit
            self.global_best = best_ant[:]

        # evaporate pheromone
        self.pheromone = [(1 - self.evaporation_rate) * t for t in self.pheromone]

        # reinforce pheromone for top-performing ants
        top_k = 5
        for i in range(min(top_k, len(self.population))):
            ant = self.population[i]
            fit = self.fitness[i]
            for idx, bit in enumerate(ant):
                if bit:
                    self.pheromone[idx] += fit / (1 + np.sum(ant))

        # clip pheromones and record history
        self.pheromone = list(np.clip(self.pheromone, self.min_pheromone, self.max_pheromone))

        self.history_global_best_fitness.append(self.global_best_fitness)
        self.history_pheromone.append(self.pheromone.copy())
        self.history_selected_features.append(self.global_best[:])
        
        selected = [self.feature_names[i] for i, bit in enumerate(self.global_best) if bit]

        # Print the best and average fitness
        print(f"Iteration {self.cur_iter + 1}: Best Fitness={self.global_best_fitness:.5f}, "
            f"Avg Fitness={avg_fitness:.5f}, Selected Features ({len(selected)}): {selected}")

        self.cur_iter += 1

    next = update_colony

    def visualize(self, save_as=None):
        """
        Heatmap of pheromone levels across features over iterations.
        """
        if not self.history_pheromone:
            print("No pheromone history to visualize.")
            return

        pheromone_matrix = np.array(self.history_pheromone)
        feature_names = self.feature_names
        iterations = list(range(1, len(self.history_pheromone) + 1))

        plt.figure(figsize=(14, 6))
        import seaborn as sns
        sns.heatmap(
            pheromone_matrix.T,
            xticklabels=iterations,
            yticklabels=feature_names,
            cmap='viridis',
            cbar_kws={'label': 'Pheromone Level'},
            linewidths=0.1,
            linecolor='gray'
        )
        plt.title('Feature Selection Patterns: Pheromone Levels Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Feature')
        plt.tight_layout()

        if save_as:
            plt.savefig(save_as, dpi=300)
        else:
            plt.show()

    def animate_optimization(self, save_as=None):
        """
        Animates two panels:
         1) Global-best fitness over iterations.
         2) Global-best feature-mask as a bar-chart with feature names.
        """
        iters = len(self.history_global_best_fitness)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios':[1,1]})

        # Panel 1: fitness curve
        ax1.set_title("Global-Best Fitness over Iterations")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Fitness")
        line, = ax1.plot([], [], lw=2, marker='o')
        ax1.set_xticks(list(range(iters)))
        ax1.set_xticklabels([str(i+1) for i in range(iters)])
        ax1.set_ylim(min(self.history_global_best_fitness)*0.9,
                     max(self.history_global_best_fitness)*1.1)

        # Panel 2: feature mask with names
        ax2.set_title("Global-Best Feature Mask")
        ax2.set_xlabel("Feature")
        ax2.set_ylabel("Selected (1) vs Not (0)")
        bar_rects = ax2.bar(self.feature_names, [0]*self.num_features)
        ax2.set_ylim(0, 1)
        plt.setp(ax2.get_xticklabels(), rotation=90, fontsize=6)

        def init():
            line.set_data([], [])
            for r in bar_rects:
                r.set_height(0)
            return [line, *bar_rects]

        def update(frame):
            x = list(range(frame+1))
            y = self.history_global_best_fitness[:frame+1]
            line.set_data(x, y)
            vec = self.history_selected_features[frame]
            for idx, r in enumerate(bar_rects):
                r.set_height(vec[idx])
                r.set_alpha(1.0 if vec[idx] else 0.3)
            ax1.set_title(f"Global-Best Fitness (Iteration {frame+1})")
            return [line, *bar_rects]

        anim = FuncAnimation(fig, update, frames=iters, init_func=init,
                             blit=True, repeat=False, interval=500)
        plt.tight_layout()

        if save_as:
            writer = writers['ffmpeg'](fps=2)
            anim.save(save_as, writer=writer)
        else:
            plt.show()


def main():
    # redirect stdout to file
    with open('aco_output.txt', 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            # 1) baseline accuracy with all features
            fs = FeatureSelection(data_path='1000_encoded_gastric_cancer_data.csv')
            full_mask = [1] * len(fs)
            baseline = round(fs.accuracy(full_mask), 5)
            print(f"\nAccuracy using all {len(fs.x.columns)} features: {baseline}")

            # 2) load data
            df = pd.read_csv('1000_encoded_gastric_cancer_data.csv')
            X, y = df.iloc[:, :-1], df.iloc[:, -1]
            feature_names = X.columns.tolist()

            # 3) run ACO
            aco = AntColonyOptimizer(
                num_agents=30,
                max_iter=50,
                train_data=X,
                train_label=y,
                default_mode=True
            )
            solution = aco.run()

            # 4) report results
            best_vec = np.array(solution.global_best)
            selected = [feature_names[i] for i, bit in enumerate(best_vec) if bit]
            best_acc = round(solution.global_best_fitness, 5)

            print("\n------------- Leader Agent ------------------------")
            print(f"Fitness: {best_acc}")
            print(f"Vector: {best_vec}")
            print(f"Selected {len(selected)} features: {selected}")
            print("---------------------------------------------------")

            # Visualize the optimization process and save
            aco.visualize(save_as="aco_feature_pattern.png")
            aco.animate_optimization(save_as="aco_output.mov")

        finally:
            sys.stdout = original_stdout


if __name__ == '__main__':
    main()