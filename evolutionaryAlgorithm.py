from random import shuffle, random, seed, gauss
from copy import copy

from tsp import TSP

def sequence_sanity_check(seq, n):
	"""Makes sure that seq is a proper sequence of all cities."""
	assert len(seq) == n
	for i in range(n):
		assert i in seq
	idxs = [i for i in range(n)]
	for x in seq:
		assert x in idxs

def crossover(ind1, ind2):
	# NOTE: ind1 and ind2 are passed by reference. Do not alter them in this function!  
	return copy(ind1) # TODO: Implement this function.

def mutate(ind):
	return ind #TODO: Implement this function

def create_random_individual(n):
	X = [i for i in range(n)]
	shuffle(X)
	return X

def convert_cost_2_fitness(X):
	"""Fitness can't be negative, and bigger has to mean better!"""
	x_max = max(X)

	return [x_max-x+1.0e-12 for x in X]

def cumsum_normalized(X):
	total = sum(X)
	out = [0.0]*len(X)

	s = 0.0
	for i in range(len(X)):
		s += X[i]/total
		out[i] = s
	return out

def pick_two_parents(cum_fitness_list):
	# NOTE: Sometimes this one picks the same parent twice. Is that chill?
	r1 = random()
	r2 = random()
	idx1 = -1
	idx2 = -1

	idxs = []
	for _ in range(2):
		r = random()
		for i in range(len(cum_fitness_list)):
			if r<=cum_fitness_list[i]:
				idxs.append(i)
				break
	assert len(idxs) == 2
	return idxs


def ea_tsp(tsp, popsize=10, maxiter=100):
	"""Evolutionary Algorithm for Traveling Salesman Problem"""
	nr_of_cities = tsp.n
	dims = tsp.dims

	cities = tsp.cities

	#Create initial population
	population = [create_random_individual(nr_of_cities) for _ in range(popsize)]

	cost_func = lambda X: tsp.calc_cost(X, should_save=False)

	for itr in range(maxiter):
		cost_list = [cost_func(individual) for individual in population]
		fitness_list = convert_cost_2_fitness(cost_list)
		cumfit_list = cumsum_normalized(fitness_list)

		next_generation = [[-1]*nr_of_cities for _ in range(popsize)]

		# Create new generation (Not necessary during the last step)
		if itr != maxiter-1:
			for p in range(popsize):

				# Pick 2 parents randomly, based on their fitness
				parent_idxs = pick_two_parents(cumfit_list)

				# Create and mutate a child.
				offspring = crossover(population[parent_idxs[0]], population[parent_idxs[1]])
				next_generation[p] = mutate(offspring)

				# Check if the output is viable
				sequence_sanity_check(next_generation[p], nr_of_cities)  #Note: Only for debugging.

			population = next_generation

	#Return best one.
	best_idx = fitness_list.index(max(fitness_list))
	return population[best_idx]

if __name__ == '__main__':
	seed(0)
	n = 50
	cities = [[gauss(0,1), gauss(0,1)] for _ in range(n)]

	tsp = TSP(cities)
	tsp.bestYet = ea_tsp(tsp)
	tsp.calc_cost(copy(tsp.bestYet), should_save=True)
	print("Final cost:", tsp.bestValYet)

	import matplotlib.pyplot as plt
	X = []
	Y = []
	for b in tsp.bestYet:
		X.append(tsp.cities[b][0])
		Y.append(tsp.cities[b][1])
	#X.append(0.0)
	#Y.append(0.0)
	X.append(X[0])
	Y.append(Y[0])
	plt.plot(X, Y)
	plt.ylabel('some numbers')
	plt.show()
