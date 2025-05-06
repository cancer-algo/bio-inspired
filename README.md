# Comparative Evaluation of Bio-Inspired Algorithms for Feature Selection in Cancer Diagnosis
Testing Bio-Inspired Algorithms on Cancer Detection Datasets - This project implements various **bio-inspired metaheuristic algorithms** for **feature selection** in a classification task using a Random Forest classifier as the evaluation model. 

## Statement of Contributions:

Ruhma implemented PSO and ACO, Dania implemented GA and GWO. Both authors contributed equally to the preprocessing, visualisation aids, experimentation and writing of the report.

## Dataset

The main dataset used is encoded on the gastric cancer data mentioned below and has 212,354 rows and 28 trainable features with a label column for diagnosis (total 29 columns). The breast cancer datatset was also used to validate the optimisation algorithms. The code repository is modular in its implementatuon and can be used on any dataset with one-hot encoded target column.

> 📂 **Gastric Cancer Dataset:** [Kaggle](https://www.kaggle.com/datasets/datasetengineer/gastric-cancer-gc-dataset)

> 📂 **Breast Cancer Dataset:** [UCI ML](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## Implemented features

- Random Forest classifier-based feature evaluator
- Particle Swarm Optimization (PSO) - `particle_swarm.py`
- Ant Colony Optimization (ACO) - `ant_colony.py`
- Genetic Algorithm (GA) - `genetic_algorithm.py`
- Grey Wolf Optimizer (GWO) - `grey_wolf_optimizer.py`
- Visualization of Iterations

## 📌 How It Works

Each optimization algorithm operates on a **binary feature mask**, where `1` indicates the feature is selected. The algorithms aim to maximize classification accuracy while minimizing the number of features.

### Model Evaluator

Implemented in `random_forest.py`:
- Uses `RandomForestClassifier` from `scikit-learn`
- Splits data into train/test and provides an `.accuracy()` method to evaluate a selected feature subset

### Bio-Inspired Optimisation Algorithms

- Wrapped around a modular base from `Py_FS`
- Optimizers iteratively improve the selected feature set
- Supports animated plots and feature mask visualizations

## 🚀 Running the Code

Ensure dependencies are installed:
```bash
pip install numpy pandas matplotlib scikit-learn seaborn Py_FS Pygame
```

Start the program with:
```bash
python run_algo.py
```

You will be prompted to choose one of the four algorithms:
1. Ant Colony Optimization (ACO)
2. Genetic Algorithm (GA)
3. Grey Wolf Optimizer (GWO)
4. Particle Swarm Optimization (PSO)

Each algorithm accepts customizable hyperparameters. After selecting an algorithm, a GUI window launches which allows the user to visualize the optimisation process of the selected features dynamically.

Alternatively, the user can run the individual algorithm by running the relevant Python files.

### Note:
1. The bio_inspired_eda.ipynb contains visualisations mentioned in the report for exploratory data analysis.
2. Sample Inputs and Outputs of all algorithm runs are stored in their respective folders. 
3. There are videos (`.mov` files for animated plots) demonstrating working algorithm. 
