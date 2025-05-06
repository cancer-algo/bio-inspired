import pandas as pd
import pygame
from grey_wolf_optimizer import GreyWolfOptimizer
from genetic_algorithm import GeneticAlgorithm
from ant_colony import AntColonyOptimizer
from particle_swarm import ParticleSwarmOptimizer
from feature_selection_game_gui import FeatureSelectionGameGUI

def get_algorithm_choice():
    print("\nSelect Bio Inspired Algorithm:")
    print("1. Ant Colony Optimization (ACO)")
    print("2. Genetic Algorithm (GA)")
    print("3. Grey Wolf Optimizer (GWO)")
    print("4. Particle Swarm Optimization (PSO)")
    print("5. Exit")
    choice = input("Enter choice (1-5): ").strip()
    return choice

def get_parameters(algo_name):

    params = {}

    while True:
        
        try:
            num_agents = int(input("Enter number of agents (max 100): ") or 30)
            if num_agents > 100:
                print("Invalid input. The number of agents cannot exceed 100.")
                continue
            params['num_agents'] = num_agents
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    
    while True:
        try:
            max_iter = int(input("Enter number of iterations (max 100): ") or 20)
            if max_iter > 100:
                print("Invalid input. The number of iterations cannot exceed 100.")
                continue
            params['max_iter'] = max_iter
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    
    if algo_name == 'GA':
        while True:
            try:
                crossover_rate = float(input("Enter crossover rate (max 1): ") or 0.8)
                if crossover_rate > 1:
                    print("Invalid input. Crossover rate cannot exceed 1.")
                    continue
                params['crossover_rate'] = crossover_rate
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        
        while True:
            try:
                mutation_rate = float(input("Enter mutation rate (max 0.5): ") or 0.1)
                if mutation_rate > 0.5:
                    print("Invalid input. Mutation rate cannot exceed 0.5.")
                    continue
                params['mutation_rate'] = mutation_rate
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    
    elif algo_name == 'ACO':
        while True:
            try:
                evaporation_rate = float(input("Enter evaporation rate (max 1): ") or 0.5)
                if evaporation_rate > 1:
                    print("Invalid input. Evaporation rate cannot exceed 1.")
                    continue
                params['evaporation_rate'] = evaporation_rate
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        
        while True:
            try:
                max_pheromone = float(input("Enter Max Pheromone (max 10): ") or 1.0)
                if max_pheromone > 10:
                    print("Invalid input. Max Pheromone cannot exceed 10.")
                    continue
                params['max_pheromone'] = max_pheromone
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    
    elif algo_name == 'PSO':
        while True:
            try:
                inertia_weight = float(input("Enter inertia weight (max 1): ") or 0.7)
                if inertia_weight > 1:
                    print("Invalid input. Inertia weight cannot exceed 1.")
                    continue
                params['inertia_weight'] = inertia_weight
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        
        while True:
            try:
                cognitive_coefficient = float(input("Enter cognitive coefficient (max 4): ") or 2.0)
                if cognitive_coefficient > 4:
                    print("Invalid input. Cognitive coefficient cannot exceed 4.")
                    continue
                params['cognitive_coefficient'] = cognitive_coefficient
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        
        while True:
            try:
                social_coefficient = float(input("Enter social coefficient (max 4): ") or 2.0)
                if social_coefficient > 4:
                    print("Invalid input. Social coefficient cannot exceed 4.")
                    continue
                params['social_coefficient'] = social_coefficient
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    
    elif algo_name == 'GWO':
        while True:
            try:
                size_penalty_weight = float(input("Enter size penalty weight (e.g., 0.01): ") or 0.01)
                params['size_penalty_weight'] = size_penalty_weight
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        
        while True:
            transfer_func = input("Enter transfer function (sigmoid/vshaped/tanh): ") or "sigmoid"
            if transfer_func in ['sigmoid', 'vshaped', 'tanh']:
                params['transfer_function'] = transfer_func
                break
            print("Invalid input. Please enter 'sigmoid', 'vshaped', or 'tanh'.")
    
    return params

def main():

    df = pd.read_csv('1000_encoded_gastric_cancer_data.csv')
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    feature_names = X.columns.tolist()

    while True:
        choice = get_algorithm_choice()
        algo = None
        algo_name = None
        algo_class = None
        params = None

        if choice == '5':
            print("Exiting program.")
            pygame.quit()
            break

        if choice == '1':
            algo_name = 'ACO'
            algo_class = AntColonyOptimizer
            params = get_parameters(algo_name)
            algo = algo_class(
                num_agents=params['num_agents'],
                max_iter=params['max_iter'],
                train_data=X,
                train_label=y,
                default_mode=True,
                evaporation_rate=params['evaporation_rate'],
                max_pheromone=params['max_pheromone']
            )
        elif choice == '2':
            algo_name = 'GA'
            algo_class = GeneticAlgorithm
            params = get_parameters(algo_name)
            algo = algo_class(
                num_agents=params['num_agents'],
                max_iter=params['max_iter'],
                train_data=X,
                train_label=y,
                default_mode=True,
                crossover_rate=params['crossover_rate'],
                mutation_rate=params['mutation_rate']
            )
        elif choice == '3':
            algo_name = 'GWO'
            algo_class = GreyWolfOptimizer
            params = get_parameters(algo_name)
            algo = algo_class(
                num_agents=params['num_agents'],
                max_iter=params['max_iter'],
                train_data=X,
                train_label=y,
                default_mode=True,
                size_penalty_weight=params['size_penalty_weight'],
                transfer_function=params['transfer_function']
            )
        elif choice == '4':
            algo_name = 'PSO'
            algo_class = ParticleSwarmOptimizer
            params = get_parameters(algo_name)
            algo = algo_class(
                num_agents=params['num_agents'],
                max_iter=params['max_iter'],
                train_data=X,
                train_label=y,
                default_mode=True,
                inertia_weight=params['inertia_weight'],
                cognitive_coefficient=params['cognitive_coefficient'],
                social_coefficient=params['social_coefficient']
            )
        else:
            print("Invalid choice! Please enter 1-5.")
            continue

        while True:
            algo.initialize()
            gui = FeatureSelectionGameGUI(algo, feature_names)
            gui.run()

            if gui.exit_gui:
                print("GUI window closed.")
            retry = input("\nDo you want to (1) return to the main menu or (2) exit: ").strip()
            while retry not in ['1', '2', '3']:
                print("\nInvalid choice! Please enter valid choice!")
                retry = input("Do you want to (1) return to the main menu or (2) exit?: ").strip()
            if retry == '2':
                print("Exiting program.")
                pygame.quit()
                return
            elif retry == '1':
                break
            

if __name__ == '__main__':
    main()