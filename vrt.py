from tsp import TSP
from utils import *

import matplotlib.pyplot as plt
from random import shuffle, randint, gauss, seed, random
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


def ns(non_fixed_cities_idxs, divided_cities, nr_of_vans, max_dist_per_van, cities, max_iter=100, grade=1):

	# Let's guess a lil bit.
	def create_guess(n, nr_of_vans):
		return [randint(0, nr_of_vans-1) for _ in range(n)]

	def mutate(X, nr_of_vans, mute_rate=0.3):
		out = copy(X)
		r = randint(0, len(X)-1)
		for i in range(len(X)):
			if i==r or random() < mute_rate:
				out[i] = randint(0, nr_of_vans-1)
		return out

	def eval(gene):
		div = deepcopy(divided_cities)
		for i in range(len(gene)):
			g = gene[i]
			idx = non_fixed_cities_idxs[i]

			div[g].append(idx)
		tsps = []
		for i in range(nr_of_vans):
			cities_copy = [cities[idx] for idx in div[i]]
			cities_copy.append(cities[0])
			tsp = TSP(cities_copy)
			tsps.append(tsp)
		for tsp in tsps:
			tsp.approximate_bounds(grade)

		total_approx = sum(tsp.bounds[1] for tsp in tsps)
		#total_approx = sum(tsp.approximate_value() for tsp in tsps)
		for tsp in tsps:
			#if tsp.approximate_value() > max_dist_per_van:
			if tsp.bounds[1] > max_dist_per_van:
				total_approx += 100 + (tsp.approximate_value()-max_dist_per_van)*(tsp.approximate_value()-max_dist_per_van)
		return total_approx

	best_guess = create_guess(len(non_fixed_cities_idxs), nr_of_vans)
	best_guess_val = eval(best_guess)

	iterations_since_update = 0

	for itr in range(max_iter):
		guess = mutate(best_guess, nr_of_vans)
		#guess = create_guess(len(non_fixed_cities_idxs), nr_of_vans)
		val = eval(guess)

		if val < best_guess_val:
			best_guess_val = val
			best_guess = guess
			iterations_since_update = 0
		else:
			iterations_since_update += 1
			if iterations_since_update > 15:
				break

	div = deepcopy(divided_cities)
	for i in range(len(best_guess)):
		g = best_guess[i]
		idx = non_fixed_cities_idxs[i]

		div[g].append(idx)
	print("here")
	return div


def eval_division_of_cities(div, cities, max_dist_per_van, nr_of_vans, grade=1):
	tsps = []
	for i in range(nr_of_vans):
		cities_copy = [cities[idx] for idx in div[i]]
		cities_copy.append(cities[0])
		tsp = TSP(cities_copy)
		tsps.append(tsp)
	for tsp in tsps:
		tsp.approximate_bounds(grade)

	total_approx = sum(tsp.approximate_value() for tsp in tsps)
	for tsp in tsps:
		if tsp.approximate_value() > max_dist_per_van:
			total_approx += 100 + (tsp.approximate_value()-max_dist_per_van)*(tsp.approximate_value()-max_dist_per_van)
	return (total_approx, tsps)


def lns(cities, nr_of_vans, max_dist_per_van, maxiter=100):
	n = len( cities )

	grade = 1

	division_of_cities = random_division_of_cities(n, nr_of_vans)
	assert len(division_of_cities) == nr_of_vans

	bestValYet = inf
	bestYet = deepcopy(division_of_cities)

	number_of_non_fixed_cities_best = 20

	for i in range(maxiter):
		non_fixed_cities_idxs = [idx for idx in range(len(cities))]
		shuffle(non_fixed_cities_idxs)
		#number_of_non_fixed_cities = 20 # TODO: optimize this dynamically.
		
		if random() < 0.25:
			diff = geo_dist(0.65)
			if random() < 0.5:
				diff = -diff
			number_of_non_fixed_cities = number_of_non_fixed_cities_best+diff
			number_of_non_fixed_cities = max(min(number_of_non_fixed_cities, len(cities)-1), 1)
		else:
			number_of_non_fixed_cities = number_of_non_fixed_cities_best
		non_fixed_cities_idxs = non_fixed_cities_idxs[:number_of_non_fixed_cities]

		# Remove the chosen cities
		division_of_cities_copy = deepcopy(division_of_cities)

		for chosen_one in non_fixed_cities_idxs:
			has_found_it = False
			for i in range(len(division_of_cities_copy)):
				for j in range(len(division_of_cities_copy[i])):
					if division_of_cities_copy[i][j] == chosen_one:
						tmp = division_of_cities_copy[i].pop(j)
						assert tmp == chosen_one
						has_found_it = True
						break
				if has_found_it:
					break

		new_div = ns(non_fixed_cities_idxs, division_of_cities_copy, nr_of_vans, max_dist_per_van, cities, grade=grade)

		(approximate_value, tsps) = eval_division_of_cities(new_div, cities, max_dist_per_van, nr_of_vans, grade=grade)
		if approximate_value < bestValYet:
			number_of_non_fixed_cities_best = number_of_non_fixed_cities
			bestValYet = approximate_value
			division_of_cities = new_div
			best_tsps = tsps
			print("Bestval:", bestValYet, number_of_non_fixed_cities_best)

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
			plt.draw()
			plt.show()


def vrt( cities, nr_of_vans, max_dist_per_van, maxiter=10000 ):
	n = len( cities )

	grade = 2

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
			cities_copy.append(cities[0])
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

			if bestValYet < 550:
				plt.close()

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
				plt.draw()
				plt.show()


if __name__ == '__main__':
	seed(1)
	n = 250
	cities = [ [gauss(0,1), gauss(0,1)] for _ in range(n) ]
	cities.insert(0, [0.0, 0.0 ])
	#vrt( cities, 5, 16.0 )
	lns(cities, 5, 16.0)


