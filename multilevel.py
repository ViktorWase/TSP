"""
This approach groups points into sub-routes.
"""

from random import randint, random, seed
from copy import copy
from math import sqrt, exp, fabs

from metaSubRoute import *

def calcDist(p1, p2):
	return sqrt(sum((p1[i]-p2[i])*(p1[i]-p2[i]) for i in range(len(p1))))

if __name__ == '__main__':
	seed(0)
	gN = 12
	gPoints = [ [random(), random()] for _ in range(gN)]

	# Create subroutes
	subroutes = [SubRoute([gPoints[3*i+1]], gPoints[3*i], gPoints[3*i+2]) for i in range(int(gN/3))]

	#biggerSubRoute = combineTwoSubRoutes(subroutes[0], subroutes[1], 0)
	#simulatedAnnealing(biggerSubRoute.points, biggerSubRoute.endPoints)

	msr = MetaSubRoute(subroutes)
	msr.optimize()
	while msr.n>1:
		print("n:", msr.n)
		msr.pickWhichSubroutesThatShouldBeCombined()
		for i in range(len(msr.subRoutes)):
			msr.subRoutes[i].smoothInternal()
	print("n:", msr.n)

	msr.optimize()

