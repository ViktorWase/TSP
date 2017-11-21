from tsp import TSP

import matplotlib.pyplot as plt


from random import shuffle, randint, gauss, seed
from math import inf
from copy import deepcopy, copy

def random_division_of_cities(nr_of_cities, nr_of_vans):
	idxs = [i for i in range(nr_of_cities)]
	shuffle(idxs)

	break_points = [randint(0, nr_of_cities) for _ in range(nr_of_vans-1)]
	break_points = sorted(break_points)

	out = [ [] for _ in range( nr_of_vans ) ]

	counter = 0
	van_counter = 0
	for idx in idxs:
		while van_counter != len(break_points) and counter >= break_points[van_counter]:
			van_counter += 1

		out[van_counter].append(idx)
		counter += 1

	return out


def mutate(division_of_cities, nr_of_cities, nr_of_vans):
	nr_of_changes = 15 # TODO: Make random

	idxs = [i for i in range(nr_of_cities)]
	shuffle(idxs)

	chosen_ones = idxs[:nr_of_changes]

	for chosen_one in chosen_ones:

		# Find it
		has_found_it = False
		for i in range(len(division_of_cities)):
			for j in range(len(division_of_cities[i])):
				if division_of_cities[i][j] == chosen_one:
					tmp = division_of_cities[i].pop(j)
					assert tmp == chosen_one
					r = randint( 0, nr_of_vans - 1)
					division_of_cities[r].append(chosen_one)
					has_found_it = True
					break
			if has_found_it:
				break
	return division_of_cities



def vrt( cities, nr_of_vans, max_dist_per_van, maxiter=10000 ):
	n = len( cities )

	grade = 1

	division_of_cities = random_division_of_cities(n, nr_of_vans)
	assert len(division_of_cities) == nr_of_vans

	bestValYet = inf
	bestYet = deepcopy(division_of_cities)

	for itr in range(maxiter):
		division_of_cities = mutate(bestYet, n, nr_of_vans)
		tsps = []
		for i in range(nr_of_vans):
			cities_copy = [cities[idx] for idx in division_of_cities[i]]
			# TODO: Add starting place here!
			tsp = TSP(cities_copy)
			tsps.append(tsp)
		for tsp in tsps:
			tsp.approximate_bounds(grade)

		total_approx = sum(tsp.approximate_value() for tsp in tsps)
		for tsp in tsps:
			if tsp.approximate_value() > max_dist_per_van:
				total_approx += 100 + (tsp.approximate_value()-max_dist_per_van)*(tsp.approximate_value()-max_dist_per_van)
		if total_approx < bestValYet:
			bestValYet = total_approx
			bestYet = deepcopy(division_of_cities)
			print(bestValYet)

			for tsp in tsps:
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


if __name__ == '__main__':
	seed(0)
	n = 60
	cities = [ [gauss(0,1), gauss(0,0.6)] for _ in range(n) ]
	vrt( cities, 5, 8.0 )


