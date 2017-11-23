from copy import deepcopy, copy
from math import sqrt, inf, exp, log
from random import gauss, randint, seed, shuffle

from utils import *

class TSP():
	"""An object for the traveling salesman problem"""
	def __init__(self, cities, starting_city=[0.0, 0.0]):
		self.cities = deepcopy(cities)
		#self.starting_city = copy(starting_city)
		self.n = len(cities)
		self.dims = len(cities[0])
		self.bestValYet = inf
		self.bestYet = [i for i in range(self.n)]

		self.bounds = [0, inf]

	def approximate_value(self):
		if self.n <= 1:
			return 0.0
		val = exp(log( 0.5*(self.bounds[1]+self.bounds[0])))
		#assert val >= self.bounds[0]-1.0e-6
		#assert val <= self.bounds[1]+1.0e-6
		return val

	def update_min(self, x):
		if x > self.bounds[0]:
			self.bounds[0] = x
		#print("max:", self.bounds[1], "min:", self.bounds[0], "gives span:", self.span())

	def update_max(self, x):
		if x < self.bounds[1]:
			self.bounds[1] = x
		#print("max:", self.bounds[1], "min:", self.bounds[0], "gives span:", self.span())

	def span(self):
		return self.bounds[1] - self.bounds[0]

	def lower_bound(self):
		vertex_2_remove = randint(0, self.n-1)
		pos = self.cities.pop(vertex_2_remove)
		
		(stuff, nodes) = self.minimum_spanning_tree()

		# Sum the edges of the minimum spanning tree. 
		s = 0.0
		for node in nodes:
			for i in range(len(node.neighs)):
				#print("hi", self.dist(node.pos, all_nodes[node.neighs[0]].pos))
				s += self.dist(node.pos, nodes[node.neighs[i]].pos)
		s /= 2.0

		self.cities.insert(vertex_2_remove, pos)

		# Find the 
		smallest = inf
		second_smallest = inf

		for j in range(self.n):
			d = self.cost_between_cities(vertex_2_remove, j)
			if d < second_smallest:
				if d < smallest:
					smallest = d
				else:
					second_smallest = d
		s += smallest  + second_smallest
		#print("here")
		self.update_min(s)

	def guess_starting(self):
		for i in range(5):
			X = [i for i in range(self.n)]
			shuffle(X)

			cost = self.calc_cost(X)

			if cost < self.bestValYet:
				self.bestValYet = cost
				self.bestYet = X
				#print(cost)

	def approximate_bounds(self, grade):

		if grade >= 0:
			self.lower_bound_min_weights()
			greed_val = self.greedy_sol(should_save=True)
			self.update_max(greed_val)
		if grade >= 1:
			even_more_improved_greed_val = self.improve_solution_using_2opts(max_iter=1000)
		if grade >= 2:
			even_more_improved_greed_val = self.improve_solution_using_2opts(max_iter=1000)
			improved_greed_val = self.improve_solution_using_straightning_swaps(max_iter=100)
			self.improve_using_point_moving()
		if grade >= 3:
			self.two_approximation(should_save=True)
			self.lower_bound()
			even_more_improved_greed_val = self.improve_solution_using_2opts()
		if grade >= 4:
			for itr in range(5*grade):
				even_more_improved_greed_val = self.improve_solution_using_2opts(max_iter=500)
				improved_greed_val = self.improve_solution_using_straightning_swaps()
				self.improve_using_point_moving()


		#self.guess_starting()
		
		self.update_max(self.bestValYet)


	def dist(self, p1, p2):
		return sqrt(sum( (p1[k]-p2[k])*(p1[k]-p2[k]) for k in range(self.dims) )) 

	def dist_sqr(self, p1, p2):
		return sum( (p1[k]-p2[k])*(p1[k]-p2[k]) for k in range(self.dims) )

	def cost_between_cities(self, i, j):
		p1 = self.cities[i]
		p2 = self.cities[j]

		return self.dist(p1, p2)
		
	def calc_cost(self, seq, should_save=False):
		s = 0.0
		for i in range(len(seq)-1):
			s += self.cost_between_cities(seq[i], seq[i+1])
		s+= self.cost_between_cities(seq[-1], seq[0])
		#s += sqrt(sum( (self.cities[seq[0]][k]-self.starting_city[k])*(self.cities[seq[0]][k]-self.starting_city[k]) for k in range(self.dims)))
		#s += sqrt(sum( (self.cities[seq[-1]][k]-self.starting_city[k])*(self.cities[seq[-1]][k]-self.starting_city[k]) for k in range(self.dims)))

		if should_save and (self.bestValYet==None or s < self.bestValYet):
			self.bestValYet = s
			bestYet = copy(seq)

		return s

	def lower_bound_min_weights(self):
		# Go through all nodes and find the 2 smallest weights, then sum them.

		chosen = [[False for _ in range(self.n)] for _ in range(self.n)]

		s = 0.0
		for i in range(self.n):
			smallest = inf
			second_smallest = inf
			best_j=-1

			for j in range(self.n):
				if i==j or chosen[i][j]:
					pass
				else:
					d = self.cost_between_cities(i, j)

					if d < second_smallest:
						if d < smallest:
							best_j = j
							smallest = d
						else:
							second_smallest = d
			s += smallest # + second_smallest
			chosen[i][best_j] = True
			chosen[best_j][i] = True

		self.update_min(s)

	def greedy_sol(self, should_save=False):
		seq = []
		unused_cities = [i for i in range(self.n)]

		#distances_from_starting_city = [ self.dist(self.cities[k], self.starting_city) for k in range(len(self.cities)) ]
		#dist_2_closest_cities = min(distances_from_starting_city)


		#current_city_idx = distances_from_starting_city.index(dist_2_closest_cities)

		current_city_idx = randint(0, self.n-1)

		seq.append(current_city_idx)
		unused_cities.pop(unused_cities.index(current_city_idx))
		#total_cost = dist_2_closest_cities

		#total_cost += sqrt(sum( (self.cities[seq[0]][k]-self.starting_city[k])*(self.cities[seq[0]][k]-self.starting_city[k]) for k in range(self.dims)))
		total_cost = 0.0
		for i in range(self.n-1):
			best_yet_idx = -1
			best_yet = inf
			for j in unused_cities:
				s = self.cost_between_cities(current_city_idx, j)
				if s < best_yet:
					best_yet = s
					best_yet_idx = j

			seq.append(best_yet_idx)
			unused_cities.pop(unused_cities.index(best_yet_idx))
			total_cost += best_yet
			current_city_idx = best_yet_idx
		total_cost += sqrt(sum( (self.cities[seq[-1]][k]-self.cities[seq[0]][k])*(self.cities[seq[-1]][k]-self.cities[seq[0]][k]) for k in range(self.dims)))
		if should_save and (self.bestValYet == None or total_cost < self.bestValYet):
			self.bestValYet = total_cost
			if should_save:
				self.bestYet = seq

		return total_cost

	def minimum_spanning_tree(self):
		# Calculates a minimum spanning tree.
		#TODO: This is O(n^3), when it can be done in linear time. Get it together tard-face!

		n = len(self.cities)

		class Node():
			"""A temporary node in a graph"""
			def __init__(self, nr, cities, starting_city_pos=[0.0, 0.0]):
				self.nr = nr
				if nr < len(cities):
					self.pos = copy(cities[nr])
				elif nr == len(cities):
					self.pos = starting_city_pos
				self.neighs = []
				self.is_in_tree = False
				self.depth = None

		all_nodes = [Node(i, self.cities) for i in range(n+1)] # The last one is the starting pos. Duh.

		all_nodes[-1].is_in_tree = True
		nodes_in_tree = [all_nodes[-1]]

		# Make it the root.
		nodes_in_tree[0].depth = 0

		total_cost = 0.0

		for i in range(len(all_nodes) - 1):
			smallest_edge_val_yet = inf
			smallest_edge_pair_yet = [-1, -1]
			for node in nodes_in_tree:
				for j in range(len(all_nodes)):
					if all_nodes[j].is_in_tree == False:
						d = self.dist_sqr(node.pos, all_nodes[j].pos)
						if d < smallest_edge_val_yet:
							smallest_edge_val_yet = d
							smallest_edge_pair_yet = [node.nr, j]
			total_cost += sqrt(smallest_edge_val_yet)
			all_nodes[smallest_edge_pair_yet[1]].is_in_tree = True
			nodes_in_tree.append(all_nodes[smallest_edge_pair_yet[1]])
			#node.neighs.append(smallest_edge_pair_yet[1])
			all_nodes[smallest_edge_pair_yet[0]].neighs.append(smallest_edge_pair_yet[1])
			nodes_in_tree[-1].neighs.append(smallest_edge_pair_yet[0])

			# Add depth
			nodes_in_tree[-1].depth = all_nodes[smallest_edge_pair_yet[0]].depth + 1

		return (nodes_in_tree, all_nodes) # TODO: These have the same info in them.

	def two_approximation(self, should_save=False):
		# TODO: Triangulate! Otherwise this one will be slooow!

		(nodes_in_tree, all_nodes) = self.minimum_spanning_tree()

		# Sum the edges of the minimum spanning tree. 
		s = 0.0
		for node in nodes_in_tree:
			for i in range(len(node.neighs)):
				#print("hi", self.dist(node.pos, all_nodes[node.neighs[0]].pos))
				s += self.dist(node.pos, all_nodes[node.neighs[i]].pos)
		s = s/2.0

		self.update_min(s)

		#Double all edges!
		for node in nodes_in_tree:
			assert len(node.neighs) >= 1
			new_list = [-1]*len(node.neighs)*2
			counter = 0
			for i in range(len(new_list)):
				new_list[i] = node.neighs[counter]
				counter += i%2
			node.neighs = new_list

		for node in nodes_in_tree:
			for neigh_nr in node.neighs:
				assert node.nr in all_nodes[neigh_nr].neighs
		"""
		#Create euler path
		paths = []
		class DoubleLinkedNode:
			def __init__(self, nr, prev):
				self.nr = nr
				self.prev = prev
				self.next = None
		while(any([ len(node.neighs)>0 for node in nodes_in_tree])):
			print("hi")
			
			# pick a node with edges
			path_start = None
			for node in nodes_in_tree:
				if len(node.neighs) > 0:
					path_start = DoubleLinkedNode(node.nr, None)
					print("NODE NR start", node.nr)
					break

			current_node = [path_start]
			while True:
				# Pick the vertex with the most edges (this doesn't have to be the case, but I think it might be fast) and don't go back unless you HAVE to.
				if len(all_nodes[current_node[0].nr].neighs) == 1:
					nr = all_nodes[current_node[0].nr].neighs[0]
					current_node[0].next = DoubleLinkedNode(nr, current_node[0])


				elif len(all_nodes[current_node[0].nr].neighs) == 0:
					# TODO: Make sure we're back at the start
					paths.append(path_start)
					break
				else:
					nr_of_neigs_list = [len(all_nodes[all_nodes[current_node[0].nr].neighs[i]].neighs) for i in range(len(all_nodes[current_node[0].nr].neighs))]

					# Remove the current from the list, and then find the max.
					neighs_list = copy(all_nodes[current_node[0].nr].neighs)
					
					# We don't want it to go back unless it haaaaaaaaaaaaas to.
					if current_node[0].prev != None and current_node[0].prev.nr in neighs_list:
						idx = neighs_list.index(current_node[0].prev.nr)
						nr_of_neigs_list[idx] = -1
					nr_of_neigs_list_max = max(nr_of_neigs_list)

					assert nr_of_neigs_list_max >= 0

					idx = nr_of_neigs_list.index(nr_of_neigs_list_max)

					nr = all_nodes[current_node[0].nr].neighs[idx]
					current_node[0].next = DoubleLinkedNode(nr, current_node[0])

				print(current_node[0].next.nr, current_node[0].nr, all_nodes[current_node[0].next.nr].neighs)


				all_nodes[current_node[0].nr].neighs.remove(current_node[0].next.nr)
				all_nodes[current_node[0].next.nr].neighs.remove(current_node[0].nr)

				current_node.append(current_node[0].next)
				current_node.pop(0)

				assert current_node[0].next == None
				assert current_node[0].prev.next.nr == current_node[0].nr

				#next_nr = all_nodes[current_node.nr].neighs[randint(0, len(all_nodes[current_node.nr].neighs)-1)]

		for path_start in paths:
			print("----")
			current_node[0] = path_start
			while current_node[0] != None:
				print("here:", current_node[0].nr)
				current_node.append(current_node[0].next)
				current_node.pop(0)

		assert len(paths) == 1
		"""

		# Find euler path using a smarter algorithm.
		path = [nodes_in_tree[0].nr]
		while True:
			current_nr = path[-1]
			neighs = all_nodes[current_nr].neighs

			if len(neighs) == 0:
				break

			# Pick a neighbour with a greater depth (if possible).
			max_depth = max( all_nodes[neig].depth for neig in neighs )
			if max_depth > all_nodes[current_nr].depth:
				tmp = [ all_nodes[neig].depth for neig in neighs ]
				next_nr = neighs[tmp.index(max_depth)]

			else: #Otherwise, just pick a random one!
				next_nr = neighs[randint(0, len(neighs)-1)]

			all_nodes[current_nr].neighs.remove(next_nr)
			all_nodes[next_nr].neighs.remove(current_nr)

			path.append(next_nr)

		assert len(path) + 1 == len(all_nodes) * 2

		tour = []
		for nr in path:
			if nr not in tour:
				tour.append(nr)

		# Rotate to make it start at starting pos
		def shift(l, n):
			return l[n:] + l[:n]
		tour = shift(tour, tour.index(self.n))
		assert tour[0] == self.n
		tour.pop(0)
		assert len(tour) == self.n 
		cost = self.calc_cost( tour, should_save=should_save )

		self.update_max(cost)

		#self.update_min(cost/2.0) # TODO: Is this correct?

	def improve_solution_using_mutation_swaps(self, max_iter=100):
		def mutate_using_swaps(X, nr_of_swaps=2):
			Y = copy(X)
			for i in range(nr_of_swaps):
				idx1 = randint(0, len(X)-1)
				idx2 = randint(0, len(X)-1)

				tmp = Y[idx1]
				Y[idx1] = Y[idx2]
				Y[idx2] = tmp
			return Y

		for itr in range(max_iter):

			seq = mutate_using_swaps(self.bestYet)

			#TODO: You don't need to eval all!
			cost = self.calc_cost(seq, should_save=True)

	def improve_using_point_moving(self, max_iter=100):
		def move_point(X, old_val):
			r1 = randint(0, len(X)-1)

			r2 = randint(0, len(X)-1)

			val = X[r1]
			X.pop(r1)

			if r2 <= r1:
				X.insert(r2, val)
			else:
				X.insert(r2-1, val)

			if self.calc_cost(X) < old_val: #TODO: We don't need to recalc all of it.
				self.bestYet = X
				self.bestValYet = val
				#print("point move", val)

	def improve_solution_using_2opts(self, max_iter=100):
		def mutate_using_2_opts():
			r1 = randint(0, self.n-1-1)

			r2 = randint(0, self.n-1-1)
			while r2 == r1 or abs(r1-r2)==1:
				r2 = randint(0, self.n-1-1)

			#Make sure that r1 is the small one
			if r2 < r1:
				tmp = r2
				r2 = r1
				r1 = tmp
			prev_part_cost = self.cost_between_cities(self.bestYet[r1], self.bestYet[r1+1]) + self.cost_between_cities(self.bestYet[r2], self.bestYet[r2+1])
			new_part_cost = self.cost_between_cities(self.bestYet[r1], self.bestYet[r2]) + self.cost_between_cities(self.bestYet[r1+1], self.bestYet[r2+1])

			if new_part_cost < prev_part_cost:
				tmp = self.bestYet[r1+1]
				self.bestYet[r1+1] = self.bestYet[r2]
				self.bestYet[r2] = tmp

				#Flip the order of the sites between r1 and r2
				mid = copy(self.bestYet[(r1+2):r2])
				mid.reverse()
				counter = 0
				for i in range(r1+2, r2):
					self.bestYet[i] = mid[counter]
					counter += 1

				val = self.calc_cost(self.bestYet, should_save=True)
				#print(val, self.bestValYet)
				#assert val < self.bestValYet
				#self.bestValYet = val
				#print("2opt", self.bestValYet)

		if self.n < 5:
			return
		for i in range(max_iter):
			mutate_using_2_opts()

	def improve_solution_using_straightning_swaps(self, max_iter=10):
		def mutate_using_2_swaps(X):
			idx_first = randint(0, len(X)-4)
			idxs1 = [idx_first, idx_first+1, idx_first+2, idx_first+3]
			idxs2 = [idx_first, idx_first+2, idx_first+1, idx_first+3]

			cost1 = sum(self.cost_between_cities(self.bestYet[idxs1[i]], self.bestYet[idxs1[i+1]]) for i in range(3))
			cost2 = sum(self.cost_between_cities(self.bestYet[idxs2[i]], self.bestYet[idxs2[i+1]]) for i in range(3))

			if cost2 < cost1:
				#self.bestValYet -= (cost1-cost2)
				tmp = self.bestYet[idx_first+1]
				self.bestYet[idx_first+1] = self.bestYet[idx_first+2]
				self.bestYet[idx_first+2] = tmp

				self.bestValYet = self.calc_cost(self.bestYet, should_save=True)
				#print(self.bestValYet)

		def mutate_using_3_swaps(X):
			idx_first = randint(0, len(X)-5)
			idxs1 = [idx_first, idx_first+1, idx_first+2, idx_first+3, idx_first+4]
			idxs2 = [idx_first, idx_first+2, idx_first+1, idx_first+3, idx_first+4]
			idxs3 = [idx_first, idx_first+1, idx_first+3, idx_first+2, idx_first+4]
			idxs4 = [idx_first, idx_first+3, idx_first+1, idx_first+2, idx_first+4]
			idxs5 = [idx_first, idx_first+3, idx_first+2, idx_first+1, idx_first+4]
			idxs6 = [idx_first, idx_first+2, idx_first+3, idx_first+1, idx_first+4]

			cost1 = sum(self.cost_between_cities(self.bestYet[idxs1[i]], self.bestYet[idxs1[i+1]]) for i in range(4))
			cost2 = sum(self.cost_between_cities(self.bestYet[idxs2[i]], self.bestYet[idxs2[i+1]]) for i in range(4))
			cost3 = sum(self.cost_between_cities(self.bestYet[idxs3[i]], self.bestYet[idxs3[i+1]]) for i in range(4))
			cost4 = sum(self.cost_between_cities(self.bestYet[idxs4[i]], self.bestYet[idxs4[i+1]]) for i in range(4))
			cost5 = sum(self.cost_between_cities(self.bestYet[idxs5[i]], self.bestYet[idxs5[i+1]]) for i in range(4))
			cost6 = sum(self.cost_between_cities(self.bestYet[idxs6[i]], self.bestYet[idxs6[i+1]]) for i in range(4))

			new_costs = [cost2, cost3, cost4, cost5, cost6]
			new_idxs = [idxs2, idxs3, idxs4, idxs5, idxs6]

			if min(new_costs) < cost1:
				best_idxs = new_idxs[new_costs.index(min(new_costs))]
				#self.bestValYet -= (cost1-cost2)

				tmps = [self.bestYet[idx] for idx in best_idxs]
				for i in range(5):
					self.bestYet[idxs1[i]] = tmps[i]

				self.bestValYet = self.calc_cost(self.bestYet, should_save=True)
				#print(self.bestValYet)

		def mutate_using_k_swaps(X, k):
			k = min(k, len(X)-2 - 1)
			if k<2:
				return
			if k == 2:
				mutate_using_2_swaps(X)
			elif k == 3:
				mutate_using_3_swaps(X)
			else:
				assert k < 10 # Otherwise we'll be here all day
				assert k < len(X)-2
				permutations = get_all_permutations_of_indexes(k)
				k_factorial = factorial(k)

				idx_first = randint(0, len(X)-k-2)
				new_idxs = [[idx_first]*(k+2) for _ in range(k_factorial)]

				counter = 0
				for perm in permutations:
					for j in range(k):
						new_idxs[counter][j+1] += perm[j]+1
					new_idxs[counter][-1] += k+1
					#print(new_idxs[counter])
					#print(idx_first)
					#print(sum(new_idxs[counter]), idx_first*(k+2) + k*(k+1)/2)
					#assert sum(new_idxs[counter]) == idx_first*(k+2) + k*(k+1)/2
					counter += 1

				new_costs =  [sum(self.cost_between_cities(self.bestYet[new_idxs[j][i]], self.bestYet[new_idxs[j][i+1]]) for i in range(k+1)) for j in range(len(new_idxs))]
				idxs1 = new_idxs.pop(0)
				cost1 = new_costs.pop(0)

				if min(new_costs) < cost1:
					best_idxs = new_idxs[new_costs.index(min(new_costs))]
					#self.bestValYet -= (cost1-cost2)

					tmps = [self.bestYet[idx] for idx in best_idxs]
					for i in range(k+2):
						self.bestYet[idxs1[i]] = tmps[i]

					val = self.calc_cost(self.bestYet, should_save=True)

					"""
					if val > self.bestValYet:
						print("oh noes")
						print(best_idxs)
						print(tmps)

					assert val < self.bestValYet
					self.bestValYet = val
					"""
					#print(self.bestValYet)
					

		#print("gere:",self.calc_cost(self.bestYet))
		for itr in range(max_iter):
			k = geo_dist(0.5)+1
			k = min(k, 6)
			mutate_using_k_swaps(self.bestYet, k)
			#mutate_using_3_swaps(self.bestYet)

if __name__ == '__main__':
	seed(1)
	n = 200
	#print("n:", n)
	cities = [[gauss(0,1), gauss(0,1)] for _ in range(n)]

	tsp = TSP(cities)

	#import cProfile
	#import re
	#cProfile.run('tsp.approximate_bounds()')

	for grade in range(6):
		tsp.approximate_bounds(grade)
		print("Grade:", grade, " val:", tsp.approximate_value())

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
