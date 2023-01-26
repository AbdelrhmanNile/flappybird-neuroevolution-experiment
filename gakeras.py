from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from copy import deepcopy
from tensorflow.keras.models import clone_model

class Brain:
    def __init__(self, keras_functional_model=None, model=None):
        if keras_functional_model is None:
            self.model = model
        else:
            self.model = keras_functional_model()

    def predict(self, x):
        return self.model.predict(x, verbose=0)

    def crossover(self, agent, crossover_rate, mutation_rate):
        child = Brain(model=clone_model(self.model))
        child_weights = deepcopy(child.model.get_weights())
        parent2_weights = deepcopy(agent.brain.model.get_weights())
        for i in range(len(child_weights)):
            for j in range(len(child_weights[i])):
                if type(child_weights[i][j]) == np.ndarray:
                    for k in range(len(child_weights[i][j])):
                        if np.random.random() < crossover_rate:
                            child_weights[i][j][k] = deepcopy(parent2_weights[i][j][k])
                        if np.random.random() < mutation_rate:
                            child_weights[i][j][k] += np.random.uniform(-5, 5)
                else:
                    if np.random.random() < crossover_rate:
                        child_weights[i][j] = deepcopy(parent2_weights[i][j])
                    if np.random.random() < mutation_rate:
                        child_weights[i][j] += np.random.uniform(-5, 5)
        child.model.set_weights(child_weights)
        return child

    def copy(self, mutation_rate):
        child = Brain(model=clone_model(self.model))
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

    # crossover by multiplying weights elementwise
    def crossover_mw(self, agent, crossover_rate, mutation_rate):
        child = Brain(model=clone_model(self.model))
        child_weights = deepcopy(child.model.get_weights())
        parent2_weights = deepcopy(agent.brain.model.get_weights())
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
        Agent_class,
        keras_functional_model,
        fitness_function,
        population_size,
    ):
        self.population = []
        self.generations = 0
        self.best_Agent = None
        self.fitness_function = fitness_function
        self.Agent_class = Agent_class
        for i in range(population_size):
            self.population.append(self.Agent_class(Brain(keras_functional_model)))

        self._evolve_methods = {
            "crossover": self._evolve_co,
            "crossover_mulwei": self._evolve_mw,
            "copy": self._evolve_copy,
        }

    def evolve(self, method, crossover_rate=None, mutation_rate=0.1, elitism_rate=0.1):
        """Evolve the population
        Available methods:
            - crossover: crossover weights of two parents
            - crossover_mulwei: crossover weights of two parents by multiplying elementwise
            - copy: copy weights of a parent and mutate them
        """
        if method not in self._evolve_methods.keys():
            raise ValueError("Invalid evolve method")
        self._evolve_methods[method](
            crossover_rate=crossover_rate, mutation_rate=mutation_rate, elitism_rate=elitism_rate
        )
        self.generations += 1
        print(f"Generation: {self.generations}")
        print(f"Best agent score: {self.best_agent.score}")

    def _evolve_co(self, crossover_rate, mutation_rate, elitism_rate):
        fitness = []
        for agent in self.population:
            fitness.append(self.fitness_function(agent))
        fitness = np.array(fitness)
        fitness = fitness / np.sum(fitness)
        #print(fitness)
        self.best_agent = deepcopy(self.population[np.argmax(fitness)])
        new_population = []
        for i in range(len(self.population)):
            if np.random.random() < elitism_rate:
                new_population.append(deepcopy(self.best_agent))
                new_population[-1].score = 0.01
                continue
            parent1 = np.random.choice(self.population, p=fitness)
            parent2 = np.random.choice(self.population, p=fitness)
            new_population.append(
                self.Agent_class(
                    parent1.brain.crossover(parent2, crossover_rate, mutation_rate)
                )
            )
        self.population.clear()
        self.population = deepcopy(new_population)
        new_population.clear()
        del new_population, fitness

    def _evolve_mw(self, crossover_rate, mutation_rate, elitism_rate):
        fitness = []
        for agent in self.population:
            fitness.append(self.fitness_function(agent))
        fitness = np.array(fitness)
        fitness = fitness / np.sum(fitness)
        self.best_agent = deepcopy(self.population[np.argmax(fitness)])
        new_population = []
        for i in range(len(self.population)):
            if np.random.random() < elitism_rate:
                new_population.append(deepcopy(self.best_agent))
                new_population[-1].score = 0.01
                continue
            parent1 = np.random.choice(self.population, p=fitness)
            parent2 = np.random.choice(self.population, p=fitness)
            new_population.append(
                self.Agent_class(
                    parent1.brain.crossover_mw(parent2, crossover_rate, mutation_rate)
                )
            )
        self.population.clear()
        self.population = deepcopy(new_population)
        new_population.clear()
        del new_population, fitness

    def _evolve_copy(self, crossover_rate, mutation_rate, elitism_rate):
        fitness = []
        for agent in self.population:
            fitness.append(self.fitness_function(agent))
        fitness = np.array(fitness)
        fitness = fitness / np.sum(fitness)
        self.best_agent = deepcopy(self.population[np.argmax(fitness)])
        new_population = []
        for i in range(len(self.population)):
            parent1 = np.random.choice(self.population, p=fitness)
            new_population.append(self.Agent_class(parent1.brain.copy(mutation_rate)))
        self.population.clear()
        self.population = deepcopy(new_population)
        new_population.clear()
        del new_population, fitness

    def get_best_agent(self):
        return deepcopy(self.best_agent)

    def get_population(self):
        return deepcopy(self.population)

    def all_dead(self):
        for agent in self.population:
            if not agent.dead:
                return False
        return True


class Agent:
    def __init__(self, brain):
        self.score = 0.01
        self.brain = brain
