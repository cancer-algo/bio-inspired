import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from Py_FS.wrapper.population_based.algorithm import Algorithm
from Py_FS.wrapper.population_based._utilities import sort_agents
from random_forest import FeatureSelection


class GeneticAlgorithm(Algorithm):

    def __init__(
            
        self
        , num_agents
        , max_iter
        , train_data
        , train_label
        , test_data=None
        , test_label=None
        , save_conv_graph=False
        , seed=0
        , default_mode=False
        , verbose=True
        , crossover_rate=0.8
        , mutation_rate=0.1

    ):
        
        super().__init__(

            num_agents=num_agents
            , max_iter=max_iter
            , train_data=train_data
            , train_label=train_label
            , test_data=test_data
            , test_label=test_label
            , save_conv_graph=save_conv_graph
            , seed=seed
            , default_mode=default_mode
            , verbose=verbose
        )

        self.algo_name = "GA"
        self.agent_name = "Chromosome"
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.history_global_best_fitness = []
        self.history_global_best_vector = []
        self.feature_names = list(train_data.columns)

        # performance settings
        self.cv_folds = 5
        self.n_jobs = -1  # use all cores

    def run(self):

        self.initialize()
        while self.cur_iter < self.max_iter:
            self.next()

        return self

    def initialize(self):

        self.num_features = self.train_data.shape[1]
        np.random.seed(self.seed)
        random.seed(self.seed)

        # cache arrays for evaluation
        self._X = self.train_data.values
        self._y = self.train_label.values

        # step 1: initialize population
        self.population = initialize_population(
            self.num_agents,
            self.num_features,
            min_features=2,
            max_features=int(self.num_features * 0.5),
        )

        self.global_best = np.zeros(self.num_features, dtype=int)
        self.global_best_fitness = -np.inf
        
        self.cur_iter = 0

    def _eval_mask(self, mask):

        # step 2: evaluate a feature subset using cross-validation accuracy
        idx = np.nonzero(mask)[0]
        if idx.size < 2:
            return 0.0
        
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )

        acc = np.mean(
            cross_val_score(
                rf,
                self._X[:, idx],
                self._y,
                cv=self.cv_folds,
                scoring="accuracy",
                n_jobs=self.n_jobs,
            )
        )

        penalty = 0.05 * (idx.size / self.num_features)  # penalize large subsets
        return acc - penalty

    def obj_function(self, population):

        # evaluate the entire population in parallel
        fitness = Parallel(n_jobs=self.n_jobs)(
            delayed(self._eval_mask)(mask) for mask in population
        )

        return np.array(fitness)

    def update_population(self):

        # step 3: evaluate fitness of current population
        self.fitness = self.obj_function(self.population)
        avg_fit = np.mean(self.fitness)

        # step 4: survival of the fittest – sort population by fitness
        self.population, self.fitness = sort_agents(
            self.population, self.fitness
        )

        if self.fitness[0] > self.global_best_fitness:
            self.global_best_fitness = self.fitness[0]
            self.global_best = self.population[0].copy()

        # step 5: elitism – retain top 10% individuals
        elite_size = max(1, int(0.1 * self.num_agents))
        elites = self.population[:elite_size].copy()

        # step 6: reproduction – fill rest of new population by cloning elites
        new_population = []
        for _ in range(self.num_agents - elite_size):
            i = np.random.randint(elite_size)
            new_population.append(elites[i].copy())

        # step 7: crossover
        for i in range(0, len(new_population), 2):
            if (
                i + 1 < len(new_population)
                and random.random() < self.crossover_rate
            ):
                pt = np.random.randint(1, self.num_features)
                a, b = new_population[i], new_population[i + 1]
                new_population[i] = np.concatenate([a[:pt], b[pt:]])
                new_population[i + 1] = np.concatenate([b[:pt], a[pt:]])

        # step 8: mutation
        for indiv in new_population:
            for j in range(self.num_features):
                if random.random() < self.mutation_rate:
                    indiv[j] = 1 - indiv[j]

            # ensure at least 2 features are selected
            if indiv.sum() < 2:
                zeros = np.where(indiv == 0)[0]
                flip = np.random.choice(zeros, 2, replace=False)
                indiv[flip] = 1

        self.population = np.vstack((elites, new_population))

        # log progress and update iteration counter
        self.history_global_best_fitness.append(self.global_best_fitness)
        self.history_global_best_vector.append(self.global_best.copy())
        selected = [
            self.feature_names[i] for i, b in enumerate(self.global_best) if b
        ]
        print(
            f"Iteration {self.cur_iter + 1}: Best Fitness={self.global_best_fitness:.5f}, "
            f"Avg Fitness={avg_fit:.5f}, Selected Features ({len(selected)}): {', '.join(selected)}"
        )

        self.cur_iter += 1

    def visualise(self, save_as=None):

        iters = len(self.history_global_best_fitness)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 1]}
        )

        ax1.plot(
            range(1, iters + 1), self.history_global_best_fitness, marker="o"
        )

        ax1.set_title("Global Best Fitness")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Fitness")

        mask = self.history_global_best_vector[-1]
        ax2.bar(self.feature_names, mask)
        ax2.set_title("Best Feature Mask")
        ax2.set_ylabel("1=Selected")

        plt.setp(ax2.get_xticklabels(), rotation=90)

        plt.tight_layout()
        if save_as:
            writer = writers["ffmpeg"](fps=2)
            anim = FuncAnimation(fig, lambda i: None)
            anim.save(save_as, writer=writer)

        else:
            plt.show()

    next = update_population


def initialize_population(pop_size, num_features, min_features=2, max_features=None):

    if max_features is None:
        max_features = num_features

    pop = []

    while len(pop) < pop_size:

        ind = np.zeros(num_features, dtype=int)
        k = random.randint(min_features, max_features)
        idx = random.sample(range(num_features), k)
        ind[idx] = 1
        pop.append(ind)

        pop = list({tuple(row) for row in pop})  # remove duplicates

    return np.array(pop, dtype=int)[:pop_size]


def main():

    with open("ga_output.txt", "w") as f:

        original_stdout = sys.stdout
        sys.stdout = f

        fs = FeatureSelection()
        X, y = fs.x, fs.y

        ga = GeneticAlgorithm(
            num_agents=60
            , max_iter=100
            , train_data=X
            , train_label=y
            , seed=42
            , default_mode=True
        )

        ga.run()

        sys.stdout = original_stdout

    ga.visualise(save_as="ga_output.mp4")

if __name__ == "__main__":
    main()
