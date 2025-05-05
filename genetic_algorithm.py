import numpy as np
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from Py_FS.wrapper.population_based.algorithm import Algorithm
from Py_FS.wrapper.population_based._utilities import sort_agents
from random_forest import FeatureSelection


class GeneticAlgorithm(Algorithm):
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
        verbose=True,
        crossover_rate=0.8,
        mutation_rate=0.1,
    ):
        # initialize parent class with parameters
        super().__init__(
            num_agents=num_agents,
            max_iter=max_iter,
            train_data=train_data,
            train_label=train_label,
            save_conv_graph=save_conv_graph,
            seed=seed,
            default_mode=default_mode,
            verbose=verbose,
        )
        # set ga-specific attributes
        self.algo_name = "GA"
        self.agent_name = "Chromosome"
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.history_global_best_fitness = []
        self.history_global_best_vector = []
        self.feature_names = self.train_data.columns.tolist()
        # initialize featureselection for tuned random forest parameters
        self.fs = FeatureSelection(
            data_path="1000_encoded_gastric_cancer_data.csv"
        )
        self.rf_params = self.fs.best_params_

    def run(self):
        # run ga for feature selection
        self.initialize()
        while self.cur_iter < self.max_iter:
            self.update_population()
        return self

    def initialize(self):
        # initialize population and parameters
        self.num_features = self.train_data.shape[1]
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.population = initialize_population(
            self.num_agents,
            self.num_features,
            min_features=2,
            max_features=int(self.num_features * 0.5),
        )
        self.global_best = [0] * self.num_features
        self.global_best_fitness = float("-inf")

    def obj_function(self, population):
        # compute fitness for each solution
        fitness = []
        for solution in population:
            selected_features = [i for i, bit in enumerate(solution) if bit]
            if not selected_features:
                fitness.append(0.0)
                continue
            X_train_selected = self.train_data.iloc[:, selected_features]
            X_test_selected = self.test_data.iloc[:, selected_features]
            rf_model = RandomForestClassifier(
                **self.rf_params, random_state=42
            )
            rf_model.fit(X_train_selected, self.train_label)
            acc = rf_model.score(X_test_selected, self.test_label)
            size_penalty = 0.01 * (np.sum(solution) / self.num_features)
            fitness.append(acc - size_penalty)
        return np.array(fitness)

    def update_population(self):
        # evolve population via selection, crossover, and mutation
        self.fitness = self.obj_function(self.population)
        avg_fitness = np.mean(self.fitness)
        self.population, self.fitness = sort_agents(
            self.population, self.fitness
        )
        if self.fitness[0] > self.global_best_fitness:
            self.global_best_fitness = self.fitness[0]
            self.global_best = self.population[0][:]

        # preserve top 10% of solutions (elitism)
        elite_size = max(1, int(0.1 * self.num_agents))
        elites = self.population[:elite_size].copy()

        # tournament selection for new population
        new_population = []
        for _ in range(self.num_agents - elite_size):
            candidates = np.random.choice(self.num_agents, size=3)
            best_idx = candidates[np.argmax(self.fitness[candidates])]
            new_population.append(self.population[best_idx][:])

        # apply single-point crossover
        for i in range(0, len(new_population), 2):
            if (
                i + 1 < len(new_population)
                and np.random.random() < self.crossover_rate
            ):
                p1, p2 = np.array(new_population[i]), np.array(
                    new_population[i + 1]
                )
                point = np.random.randint(1, self.num_features - 1)
                c1 = np.concatenate([p1[:point], p2[point:]])
                c2 = np.concatenate([p2[:point], p1[point:]])
                new_population[i], new_population[i + 1] = c1, c2

        # apply bit-flip mutation
        for i in range(len(new_population)):
            for j in range(self.num_features):
                if np.random.random() < self.mutation_rate:
                    new_population[i][j] = 1 - new_population[i][j]
            if np.array_equal(new_population[i], self.global_best):
                flip_idx = np.random.randint(self.num_features)
                new_population[i][flip_idx] = 1 - new_population[i][flip_idx]

        self.population = np.vstack((elites, new_population))
        self.history_global_best_fitness.append(self.global_best_fitness)
        self.history_global_best_vector.append(self.global_best.copy())
        selected = [
            self.feature_names[i]
            for i, bit in enumerate(self.global_best)
            if bit
        ]
        print(
            f"Iteration {self.cur_iter + 1}: Best Fitness={self.global_best_fitness:.5f}, "
            f"Avg Fitness={avg_fitness:.5f}, Selected Features ({len(selected)}): {', '.join(selected)}"
        )
        self.cur_iter += 1

    def visualize(self, save_as=None):
        # visualize fitness and feature mask evolution
        iters = len(self.history_global_best_fitness)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1]}
        )

        ax1.set_title("Global-Best Fitness over Iterations")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Fitness")
        (line,) = ax1.plot([], [], lw=2, marker="o")
        ax1.set_xticks(list(range(iters)))
        ax1.set_xticklabels([str(i + 1) for i in range(iters)])
        ax1.set_ylim(
            min(self.history_global_best_fitness) * 0.9,
            max(self.history_global_best_fitness) * 1.1,
        )

        ax2.set_title("Global-Best Feature Mask")
        ax2.set_xlabel("Feature")
        ax2.set_ylabel("Selected (1) vs Not (0)")
        bar_rects = ax2.bar(self.feature_names, [0] * self.num_features)
        ax2.set_ylim(0, 1)
        plt.setp(
            ax2.get_xticklabels(),
            rotation=90,
            fontsize=8 if self.num_features <= 30 else 6,
        )

        def init():
            line.set_data([], [])
            for r in bar_rects:
                r.set_height(0)
            return [line, *bar_rects]

        def update(frame):
            x = list(range(frame + 1))
            y = self.history_global_best_fitness[: frame + 1]
            line.set_data(x, y)
            vec = self.history_global_best_vector[frame]
            for idx, r in enumerate(bar_rects):
                r.set_height(vec[idx])
                r.set_alpha(1.0 if vec[idx] else 0.3)
            ax1.set_title(f"Global-Best Fitness (Iteration {frame + 1})")
            return [line, *bar_rects]

        anim = FuncAnimation(
            fig,
            update,
            frames=iters,
            init_func=init,
            blit=True,
            repeat=False,
            interval=500,
        )
        plt.tight_layout()
        if save_as:
            writer = writers["ffmpeg"](fps=2)
            anim.save(save_as, writer=writer)
        else:
            plt.show()


def initialize_population(
    pop_size, num_features, min_features=2, max_features=None
):
    # generate initial population of feature masks
    if max_features is None:
        max_features = num_features
    population = []
    for _ in range(pop_size):
        individual = np.zeros(num_features, dtype=int)
        num_selected = random.randint(min_features, max_features)
        selected_indices = random.sample(range(num_features), num_selected)
        individual[selected_indices] = 1
        population.append(individual)
    return np.array(population)


def main():
    # run ga for feature selection
    with open("ga_output.txt", "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            df = pd.read_csv("1000_encoded_gastric_cancer_data.csv")
            X, y = df.iloc[:, :-1], df.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )

            # compute baseline accuracy with all features
            fs = FeatureSelection(
                data_path="1000_encoded_gastric_cancer_data.csv"
            )
            full_mask = [1] * len(fs)
            baseline = round(fs.accuracy(full_mask), 5)
            print(
                f"\nAccuracy using all {len(fs.x.columns)} features: {baseline}"
            )

            # initialize and run ga
            ga = GeneticAlgorithm(
                num_agents=30,
                max_iter=50,
                train_data=X_train,
                train_label=y_train,
                test_data=X_test,
                test_label=y_test,
                default_mode=True,
            )
            ga.run()

            # report final results
            best_vec = np.array(ga.global_best)
            selected = [
                ga.feature_names[i] for i, bit in enumerate(best_vec) if bit
            ]
            best_acc = round(ga.global_best_fitness, 5)
            print("\n------------- Leader Agent ------------------------")
            print(f"Fitness: {best_acc}")
            print(f"Vector: {best_vec}")
            print(f"Selected {len(selected)} features: {', '.join(selected)}")
            print("---------------------------------------------------")

            # save visualizations
            fs.plot_feature_importance(
                best_vec, save_path="ga_feature_importance.png"
            )
            ga.visualize(save_as="ga_output.mov")

        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
