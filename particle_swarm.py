import numpy as np
import pandas as pd
import sys
from Py_FS.wrapper.population_based.algorithm import Algorithm
from Py_FS.wrapper.population_based._utilities import sort_agents
from Py_FS.wrapper.population_based._transfer_functions import get_trans_function
from random_forest import FeatureSelection
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

class ParticleSwarmOptimizer(Algorithm):
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
        self.algo_name = 'PSO'
        self.agent_name = 'Particle'
        self.trans_function = get_trans_function('s')

        if hasattr(train_data, 'columns'):
            self.feature_names = list(train_data.columns)
        else:
            self.feature_names = [str(i) for i in range(self.num_features)]

    def initialize(self):
        super().initialize()
        self.global_best = [0] * self.num_features
        self.global_best_fitness = float('-inf')
        self.local_best = [[0] * self.num_features for _ in range(self.num_agents)]
        self.local_best_fitness = [float('-inf')] * self.num_agents
        self.velocity = [[0.0] * self.num_features for _ in range(self.num_agents)]
        self.weight = 1.0

        # History buffers for visualization
        self.history_global_best_fitness = []
        self.history_global_best_vector  = []

    def update_swarm(self):
        header = f"\n{'='*80}\nIteration - {self.cur_iter + 1}\n{'='*80}"
        self.print(header)

        # linearly decreasing inertia weight
        self.weight = 1.0 - (self.cur_iter / self.max_iter)

        # Phase 1: update velocities
        for i in range(self.num_agents):
            for j in range(self.num_features):
                r1, r2 = np.random.random(2)
                self.velocity[i][j] = (
                    self.weight * self.velocity[i][j]
                    + r1 * (self.local_best[i][j] - self.population[i][j])
                    + r2 * (self.global_best[j] - self.population[i][j])
                )

        # Phase 2: update positions using transfer function
        for i in range(self.num_agents):
            for j in range(self.num_features):
                trans_val = self.trans_function(self.velocity[i][j])
                self.population[i][j] = int(np.random.random() < trans_val)

        # evaluate and sort agents
        self.fitness = self.obj_function(self.population, self.training_data)
        self.population, self.fitness = sort_agents(self.population, self.fitness)

        # update personal and global bests
        for idx, fit in enumerate(self.fitness):
            if fit > self.local_best_fitness[idx]:
                self.local_best_fitness[idx] = fit
                self.local_best[idx] = self.population[idx][:]
            if fit > self.global_best_fitness:
                self.global_best_fitness = fit
                self.global_best = self.population[idx][:]

        # ensure binary output and increment iteration
        self.global_best = [int(x) for x in self.global_best]

        # Record history for visualization
        self.history_global_best_fitness.append(self.global_best_fitness)
        self.history_global_best_vector.append(self.global_best.copy())

        self.cur_iter += 1
    
    next = update_swarm


    def visualize(self, save_as=None):
        """
        Animates two panels:
         1) Global-best fitness over iterations (categorical iterations).
         2) Global-best feature-mask as a bar-chart with feature names.
        Saves to pso_output.mov
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
            # fitness curve
            x = list(range(frame+1))
            y = self.history_global_best_fitness[:frame+1]
            line.set_data(x, y)
            # bar heights
            vec = self.history_global_best_vector[frame]
            for idx, r in enumerate(bar_rects):
                r.set_height(vec[idx])
                # highlight selected
                r.set_alpha(1.0 if vec[idx] else 0.3)
            # annotate iteration
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
    with open('pso_output.txt', 'w') as f:
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

            # 3) run PSO
            pso = ParticleSwarmOptimizer(
                num_agents=30,
                max_iter=20,
                train_data=X,
                train_label=y,
                default_mode=True
            )
            solution = pso.run()

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
            pso.visualize(save_as="pso_output.mov")

        finally:
            sys.stdout = original_stdout


if __name__ == '__main__':
    main()
