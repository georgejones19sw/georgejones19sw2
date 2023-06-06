import numpy as np

class GeneticAlgorithm:
    def __init__(self, target_string, population_size=100, mutation_rate=0.01):
        self.target_string = target_string
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = ''.join(np.random.choice(list(self.target_string)) for _ in range(len(self.target_string)))
            population.append(chromosome)
        return population

    def fitness(self, chromosome):
        return sum(c1 == c2 for c1, c2 in zip(chromosome, self.target_string))

    def selection(self):
        fitness_scores = [self.fitness(chromosome) for chromosome in self.population]
        probabilities = np.array(fitness_scores) / sum(fitness_scores)
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size, p=probabilities)
        return [self.population[index] for index in selected_indices]

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1))
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutation(self, chromosome):
        mutated_chromosome = ""
        for char in chromosome:
            if np.random.random() < self.mutation_rate:
                mutated_chromosome += np.random.choice(list(self.target_string))
            else:
                mutated_chromosome += char
        return mutated_chromosome

    def evolve(self):
        new_population = []

        selected_population = self.selection()

        while len(new_population) < self.population_size:
            parent1 = np.random.choice(selected_population)
            parent2 = np.random.choice(selected_population)
            child1, child2 = self.crossover(parent1, parent2)
            mutated_child1 = self.mutation(child1)
            mutated_child2 = self.mutation(child2)
            new_population.append(mutated_child1)
            new_population.append(mutated_child2)

        self.population = new_population

    def run(self, max_generations=100):
        generation = 0
        best_fitness = 0
        best_chromosome = ""

        while generation < max_generations and best_fitness < len(self.target_string):
            generation += 1
            self.evolve()
            best_chromosome = max(self.population, key=self.fitness)
            best_fitness = self.fitness(best_chromosome)

            print(f"Generation: {generation}, Best Fitness: {best_fitness}, Best Chromosome: {best_chromosome}")

# Example usage:
target = "Hello, World!"
ga = GeneticAlgorithm(target)
ga.run()
