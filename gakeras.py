from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from copy import deepcopy


class Brain:
    def __init__(self, keras_functional_model=None, model=None):
        if keras_functional_model is None:
            self.model = model
        else:
            self.model = keras_functional_model()

    def predict(self, x):
        return self.model.predict(x, verbose=0)

    def crossover(self, individual, crossover_rate, mutation_rate):
        child = Brain(model=deepcopy(self.model))
        child_weights = deepcopy(child.model.get_weights())
        parent2_weights = deepcopy(individual.brain.model.get_weights())
        for i in range(len(child_weights)):
            for j in range(len(child_weights[i])):
                if type(child_weights[i][j]) == np.ndarray:
                    for k in range(len(child_weights[i][j])):
                        if np.random.random() < crossover_rate:
                            child_weights[i][j][k] = parent2_weights[i][j][k]
                        if np.random.random() < mutation_rate:
                            child_weights[i][j][k] += np.random.normal(0, 1)
                else:
                    if np.random.random() < crossover_rate:
                        child_weights[i][j] = parent2_weights[i][j]
                    if np.random.random() < mutation_rate:
                        child_weights[i][j] += np.random.normal(0, 1)
        child.model.set_weights(child_weights)
        return child

    def copy(self, mutation_rate):
        child = Brain(model=deepcopy(self.model))
        child_weights = deepcopy(child.model.get_weights())
        for i in range(len(child_weights)):
            for j in range(len(child_weights[i])):
                if type(child_weights[i][j]) == np.ndarray:
                    for k in range(len(child_weights[i][j])):
                        if np.random.random() < mutation_rate:
                            child_weights[i][j][k] += np.random.normal(0, 1)
                else:
                    if np.random.random() < mutation_rate:
                        child_weights[i][j] += np.random.normal(0, 1)
        child.model.set_weights(child_weights)
        return child

    # crossover by multiplying weights
    def crossover_mw(self, individual, crossover_rate, mutation_rate):
        child = Brain(deepcopy(self.model))
        child_weights = deepcopy(child.model.get_weights())
        parent2_weights = deepcopy(individual.model.get_weights())
        for i in range(len(child_weights)):
            for j in range(len(child_weights[i])):
                if type(child_weights[i][j]) == np.ndarray:
                    for k in range(len(child_weights[i][j])):
                        if np.random.random() < crossover_rate:
                            child_weights[i][j][k] = (
                                child_weights[i][j][k] * parent2_weights[i][j][k]
                            )
                        if np.random.random() < mutation_rate:
                            child_weights[i][j][k] += np.random.normal(0, 1)
                else:
                    if np.random.random() < crossover_rate:
                        child_weights[i][j] = (
                            child_weights[i][j] * parent2_weights[i][j]
                        )
                    if np.random.random() < mutation_rate:
                        child_weights[i][j] += np.random.normal(0, 1)
        child.model.set_weights(child_weights)
        return child


class Population:
    def __init__(
        self,
        Individual_class,
        keras_functional_model,
        fitness_function,
        population_size,
    ):
        self.population = []
        self.generations = 0
        self.best_individual = None
        self.fitness_function = fitness_function
        self.Individual_class = Individual_class
        for i in range(population_size):
            self.population.append(self.Individual_class(Brain(keras_functional_model)))

    def evolve(self, crossover_rate, mutation_rate):
        fitness = []
        for individual in self.population:
            fitness.append(self.fitness_function(individual))
        fitness = np.array(fitness)
        fitness = fitness / np.sum(fitness)
        self.best_individual = self.population[np.argmax(fitness)]
        new_population = []
        for i in range(len(self.population)):
            parent1 = np.random.choice(self.population, p=fitness)
            parent2 = np.random.choice(self.population, p=fitness)
            new_population.append(
                self.Individual_class(
                    parent1.brain.crossover(parent2, crossover_rate, mutation_rate)
                )
            )
        self.population = new_population
        self.generations += 1
        print(f"generation {self.generations}")
        print(f"best score {self.best_individual.score}")

    def evolve_mw(self, crossover_rate, mutation_rate):
        fitness = []
        for individual in self.population:
            fitness.append(self.fitness_function(individual))
        fitness = np.array(fitness)
        fitness = fitness / np.sum(fitness)
        self.best_individual = self.population[np.argmax(fitness)]
        new_population = []
        for i in range(len(self.population)):
            parent1 = np.random.choice(self.population, p=fitness)
            parent2 = np.random.choice(self.population, p=fitness)
            new_population.append(
                parent1.crossover_mw(parent2, crossover_rate, mutation_rate)
            )
        self.population = new_population

    def evolve_copy(self, mutation_rate):
        fitness = []
        for individual in self.population:
            fitness.append(self.fitness_function(individual))
        fitness = np.array(fitness)
        fitness = fitness / np.sum(fitness)
        self.best_individual = self.population[np.argmax(fitness)]
        new_population = []
        for i in range(len(self.population)):
            # parent1 = np.random.choice(self.population, p=fitness)
            parent1 = self.population[i]
            new_population.append(
                self.Individual_class(parent1.brain.copy(mutation_rate))
            )
        self.population = new_population

    def get_best_individual(self):
        return self.best_individual

    def get_population(self):
        return self.population

    def all_dead(self):
        for individual in self.population:
            if not individual.dead:
                return False
        return True


class Individual:
    def __init__(self, brain):
        self.score = 0.01
        self.brain = brain
