"""
This approach groups points into sub-routes.
"""

from random import randint, random, seed
from copy import copy
from math import sqrt, exp, fabs

from metaSubRoute import *

def calcDist(p1, p2):
	return sqrt(sum((p1[i]-p2[i])*(p1[i]-p2[i]) for i in range(len(p1))))

def multiLevelOptimization(points, iters=5):
	# Create subroutes
	subroutes = [SubRoute([points[3*i+1]], points[3*i], points[3*i+2]) for i in range(int(len(points)/3))]

	msr = MetaSubRoute(subroutes)

	# TODO: Replace this with a starting heuristic.
	print("TOTAL DIST:", msr.getTotalDist(), msr.getProperTotalDist())
	msr.optimize()
	#print("TOTAL DIST:", msr.getTotalDist(), msr.getProperTotalDist())

	for itr in range(iters):
		# Combine subroutes and to internal optimization.
		#assert fabs(msr.getProperTotalDist() - msr.getTotalDist()) < 1.0e-10
		while msr.n>1:
			#print("COMBINE n:", msr.n, "TOTAL DIST:", msr.getTotalDist(), msr.getProperTotalDist())
			msr.pickWhichSubroutesThatShouldBeCombined()
			#assert fabs(msr.getProperTotalDist() - msr.getTotalDist()) < 1.0e-10
			for i in range(len(msr.subRoutes)):
				msr.subRoutes[i].smoothInternal()
				#assert fabs(msr.getProperTotalDist() - msr.getTotalDist()) < 1.0e-10

		# Divide subroutes and to external optimization
		while msr.n<int(gN/5): #TODO: 5 is arbitraty. Tune.
			msr.divideSubroutes()
			#assert fabs(msr.getProperTotalDist() - msr.getTotalDist()) < 1.0e-10
			msr.optimize()
			#assert fabs(msr.getProperTotalDist() - msr.getTotalDist()) < 1.0e-10
			#print("DIVIDE n:", msr.n, "TOTAL DIST:", msr.getTotalDist(), msr.getProperTotalDist())
		#print("n:", msr.n)

		print("TOTAL DIST:", msr.getTotalDist(), msr.getProperTotalDist())
		assert fabs(msr.getProperTotalDist() - msr.getTotalDist()) < 1.0e-10
	return msr.getTotalDist()

if __name__ == '__main__':
	seed(0)
	gN = 1200
	gPoints = [ [random(), random()] for _ in range(gN)]

	import cProfile
	import re
	multiLevelOptimization(gPoints)
	#cProfile.run("multiLevelOptimization(gPoints)")

	
	

