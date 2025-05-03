# Bio-Inspired Feature Selection for Classification
Testing Bio-Inspired Algorithms on Cancer Detection Datasets

This project implements various **bio-inspired metaheuristic algorithms** for **feature selection** in a classification task using a Random Forest classifier as the evaluation model. 

The dataset used is encoded on the gastric cancer data mentioned below and has 212,354 rows and 28 trainable features with a label column for diagnosis (total 29 columns).
> 📂 **Dataset:** [Gastric Cancer (GC) Dataset on Kaggle](https://www.kaggle.com/datasets/datasetengineer/gastric-cancer-gc-dataset)

## Implemented features

- Random Forest classifier-based feature evaluator
- Particle Swarm Optimization (PSO)
- Ant Colony Optimization (ACO)
- Genetic Algorithm (GA)
- Grey Wolf Optimizer (GWO)
- Visualizations

## 📌 How It Works

Each optimization algorithm operates on a **binary feature mask**, where `1` indicates the feature is selected. The algorithms aim to maximize classification accuracy while minimizing the number of features.

### Model Evaluator

Implemented in `random_forest.py`:
- Uses `RandomForestClassifier` from `scikit-learn`
- Splits data into train/test and provides an `.accuracy()` method to evaluate a selected feature subset

### PSO & ACO

- Wrapped around a modular base from `Py_FS`
- Optimizers iteratively improve the selected feature set
- Supports animated plots and feature mask visualizations

## Running the Code

Ensure dependencies are installed:
```bash
pip install numpy pandas matplotlib scikit-learn seaborn Py_FS


