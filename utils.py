from itertools import permutations
from random import random

def factorial(n):
	if n == 1:
		return 1
	else:
		return n*factorial(n-1)


def get_all_permutations_of_indexes(n):
	assert n > 0
	assert n < 10 # Mostly because anything bigger than 10 will take a lot of time and memory.
	return permutations(range(n))
	#n_factorial = factorial(n)
	#permutations = [[0]*n for _ in range(n_factorial)]

def geo_dist(p):
	x = 1
	while(random() < p):
		x+=1
	return x