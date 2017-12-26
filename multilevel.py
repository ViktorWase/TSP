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
	gN = 120
	gPoints = [ [random(), random()] for _ in range(gN)]

	# Create subroutes
	subroutes = [SubRoute([gPoints[3*i+1]], gPoints[3*i], gPoints[3*i+2]) for i in range(int(gN/3))]

	msr = MetaSubRoute(subroutes)

	# TODO: Replace this with a starting heuristic.
	print("TOTAL DIST:", msr.getTotalDist(), msr.getProperTotalDist())
	msr.optimize()
	print("TOTAL DIST:", msr.getTotalDist(), msr.getProperTotalDist())
	msr.pickWhichSubroutesThatShouldBeCombined()
	print("TOTAL DIST:", msr.getTotalDist(), msr.getProperTotalDist())
	for i in range(len(msr.subRoutes)):
		msr.subRoutes[i].smoothInternal()
	print("TOTAL DIST:", msr.getTotalDist(), msr.getProperTotalDist())


	"""
	for itr in range(4):
		# Combine subroutes and to internal optimization.
		while msr.n>1:
			print("n:", msr.n)
			msr.pickWhichSubroutesThatShouldBeCombined()
			print("starting optimization.")
			for i in range(len(msr.subRoutes)):
				msr.subRoutes[i].smoothInternal()
		print("n:", msr.n)

		# Divide subroutes and to external optimization
		while msr.n<int(gN/5): #TODO: 5 is arbitraty. Tune.
			msr.divideSubroutes()
			print("starting optimization.")
			msr.optimize()
			print("n:", msr.n)
		#print("n:", msr.n)

		print("TOTAL DIST:", msr.getTotalDist(), msr.getProperTotalDist())
	"""

