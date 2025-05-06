import numpy as np
import random
from joblib import Parallel, delayed
from Py_FS.wrapper.population_based.algorithm import Algorithm
from Py_FS.wrapper.population_based._transfer_functions import get_trans_function
from scipy.spatial.distance import hamming
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sys
from contextlib import redirect_stdout
from random_forest import FeatureSelection

class GreyWolfOptimizer(Algorithm):

    def __init__(
            
        self
        , num_agents
        , max_iter
        , train_data
        , train_label
        , test_data=None
        , test_label=None
        , seed=0
        , verbose=True
        , default_mode=False
        , n_estimators=300
        , cv_folds=5

    ):
        super().__init__(
            num_agents=num_agents
            , max_iter=max_iter
            , train_data=train_data
            , train_label=train_label
            , test_data=test_data
            , test_label=test_label
            , seed=seed
            , default_mode=default_mode
            , verbose=verbose
        )

        self.algo_name = "GWO"
        self.agent_name = "Wolf"
        self.feature_names = list(train_data.columns)
        self.trans_function = get_trans_function("s")

        # hyperparameters
        self.n_estimators = n_estimators
        self.cv_folds = cv_folds

        # tracking state
        self.cur_iter = 0
        self.stagnation_count = 0
        self.prev_best_fitness = -np.inf
        self.history_global_best_fitness = []
        self.history_global_best_vector = []

    def initialize(self):

        # set random seeds
        np.random.seed(self.seed)
        random.seed(self.seed)

        # cache raw arrays for fast slicing
        self._X = self.train_data.values
        self._y = self.train_label.values

        # initialize wolf pack
        self.num_features = self._X.shape[1]
        self.population = initialize_population(
            self.num_agents,
            self.num_features,
            min_features=1,
            max_features=int(0.8 * self.num_features),
        )

        # initialize alpha, beta, delta wolves
        self.alpha_pos = np.zeros(self.num_features, dtype=int)
        self.alpha_score = -np.inf
        self.beta_pos  = np.zeros(self.num_features, dtype=int)
        self.beta_score = -np.inf
        self.delta_pos  = np.zeros(self.num_features, dtype=int)
        self.delta_score = -np.inf

    def _eval_mask(self, mask):

        # evaluate one solution using random forest + cross-validation
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return 0.0

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=-1,
        )

        acc = np.mean(
            cross_val_score(
                rf,
                self._X[:, idx],
                self._y,
                cv=self.cv_folds,
                scoring="accuracy",
                n_jobs=-1,
            )
        )

        penalty   = 0.01 * (idx.size / self.num_features)  # penalize large subsets
        diversity = (
            hamming(mask, self.alpha_pos) * 0.05 if self.cur_iter > 0 else 0.0
        )  # encourage diversity from alpha

        return acc - penalty + diversity

    def obj_function(self, population):

        # evaluate all wolves in parallel
        return np.array(
            Parallel(n_jobs=-1)(
                delayed(self._eval_mask)(mask) for mask in population
            )
        )

    def update_pack(self):

        # step 1: evaluate fitness
        fitness = self.obj_function(self.population)
        avg_fit = fitness.mean()

        # step 2: identify alpha, beta, delta
        idxs = np.argsort(fitness)[::-1]
        for i in idxs:
            f   = fitness[i]
            pos = self.population[i].copy()

            if f > self.alpha_score:
                self.delta_score, self.delta_pos = self.beta_score,  self.beta_pos.copy()
                self.beta_score,  self.beta_pos  = self.alpha_score, self.alpha_pos.copy()
                self.alpha_score, self.alpha_pos = f, pos
            elif f > self.beta_score and hamming(pos, self.alpha_pos) > 0.1:
                self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                self.beta_score,  self.beta_pos  = f, pos
            elif f > self.delta_score and hamming(pos, self.alpha_pos) > 0.1:
                self.delta_score, self.delta_pos = f, pos

        # step 3: update position of each wolf based on alpha, beta, delta
        a = 2 * (1 - (self.cur_iter / self.max_iter) ** 2)
        new_pop = []

        for wolf in self.population:
            upd = wolf.copy()

            for j in range(self.num_features):
                def move(leader):
                    r1, r2 = random.random(), random.random()
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = abs(C * leader[j] - wolf[j])
                    return leader[j] - A * D

                val = (move(self.alpha_pos) +
                       move(self.beta_pos)  +
                       move(self.delta_pos)) / 3.0

                thr = self.trans_function(val) * (1 - self.cur_iter / self.max_iter)
                upd[j] = 1 if random.random() < thr else 0

                # apply small mutation
                if random.random() < 0.05:
                    m = random.randrange(self.num_features)
                    upd[m] = 1 - upd[m]

            new_pop.append(upd)

        self.population = np.array(new_pop, dtype=int)

        # step 4: update global best
        self.global_best         = self.alpha_pos.copy()
        self.global_best_fitness = self.alpha_score
        self.history_global_best_fitness.append(self.global_best_fitness)
        self.history_global_best_vector.append(self.global_best.copy())

        # step 5: detect stagnation
        if abs(self.global_best_fitness - self.prev_best_fitness) < 1e-4:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        self.prev_best_fitness = self.global_best_fitness

        # step 6: reset if stuck too long
        if self.stagnation_count > 10:
            self.population = initialize_population(
                self.num_agents,
                self.num_features,
                min_features=1,
                max_features=int(0.8 * self.num_features),
            )
            self.stagnation_count = 0

        # log progress
        sel = [self.feature_names[i] for i, b in enumerate(self.global_best) if b]
        print(
            f"Iteration {self.cur_iter + 1}: "
            f"Best Fitness={self.global_best_fitness:.5f}, "
            f"Avg Fitness={avg_fit:.5f}, "
            f"Selected Features ({len(sel)}): {', '.join(sel)}"
        )

        self.cur_iter += 1

    next = update_pack

    def run(self):
        self.initialize()
        while self.cur_iter < self.max_iter and self.stagnation_count <= 10:
            self.next()
        return self


def initialize_population(pop_size, num_features, min_features=1, max_features=None):

    # generate a population of binary masks

    if max_features is None:
        max_features = num_features

    pop = []

    while len(pop) < pop_size:

        ind = np.zeros(num_features, dtype=int)
        k = random.randint(min_features, max_features)
        idx = random.sample(range(num_features), k)
        ind[idx] = 1

        pop.append(ind)
        pop = list({tuple(x) for x in pop})  # remove duplicates

    return np.array(pop, dtype=int)[:pop_size]


def main():

    fs = FeatureSelection()

    X, y = fs.x, fs.y

    gwo = GreyWolfOptimizer(
        num_agents=25
        , max_iter=40
        , train_data=X
        , train_label=y
        , seed=42
        , default_mode=True
    )


    with open("gwo_output.txt", "w") as f:
        with redirect_stdout(f):
            gwo.run()


if __name__ == "__main__":
    main()
