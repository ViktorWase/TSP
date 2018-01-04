"""
This approach groups points into sub-routes.
"""

from random import randint, random, seed, gauss
from copy import copy
from math import sqrt, exp, fabs

from metaSubRoute import *

def calcDist(p1, p2):
	return sqrt(sum((p1[i]-p2[i])*(p1[i]-p2[i]) for i in range(len(p1))))

def calcDistSqr(p1, p2):
	return sum((p1[i]-p2[i])*(p1[i]-p2[i]) for i in range(len(p1)))

def multiLevelOptimization(points, iters=25):
	# Create subroutes
	subroutes = [SubRoute([points[3*i+1]], points[3*i], points[3*i+2]) for i in range(int(len(points)/3))]

	msr = MetaSubRoute(subroutes)

	# These are only for debugging.
	gainFromExternal = 0.0
	gainFromInternal = 0.0

	# TODO: Replace this with a starting heuristic.
	print("TOTAL DIST:", msr.getTotalDist(), msr.getProperTotalDist())
	msr.optimize()
	#print("TOTAL DIST:", msr.getTotalDist(), msr.getProperTotalDist())


	dist = msr.getTotalDist() #THIS IS ONLY FOR DEBUGGING

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
		assert fabs(msr.getProperTotalDist() - msr.getTotalDist()) < 1.0e-10

		newDist = msr.getTotalDist() #THIS IS ONLY FOR DEBUGGING
		gainFromInternal += dist-newDist
		dist = newDist

		# Divide subroutes and to external optimization
		while msr.n<int(gN/5): #TODO: 5 is arbitraty. Tune.
			msr.divideSubroutes()
			#assert fabs(msr.getProperTotalDist() - msr.getTotalDist()) < 1.0e-10
			msr.optimize()
			#assert fabs(msr.getProperTotalDist() - msr.getTotalDist()) < 1.0e-10
			#print("DIVIDE n:", msr.n, "TOTAL DIST:", msr.getTotalDist(), msr.getProperTotalDist())
		#print("n:", msr.n)
		newDist = msr.getTotalDist() #THIS IS ONLY FOR DEBUGGING
		gainFromExternal += dist-newDist
		dist = newDist

		print("internal:", gainFromInternal, "external:", gainFromExternal)

		print("TOTAL DIST:", msr.getTotalDist(), msr.getProperTotalDist())
		print("")
		assert fabs(msr.getProperTotalDist() - msr.getTotalDist()) < 1.0e-10
	return msr

def initialGuess(points):
	n = len(points)

	remainingPoints = deepcopy(points)

	out = []
	rndPoint = randint(0, n-1)
	out.append(remainingPoints.pop(rndPoint))

	for i in range(n-1):
		bestDistSqr = inf
		bestIdx = -1
		for j in range(20): #TODO: Tune 10 or do something clever.
			r = randint(0, len(remainingPoints)-1)
			p = remainingPoints[r]
			distSqr = calcDistSqr(p, out[-1])

			if distSqr < bestDistSqr:
				bestDistSqr = distSqr
				bestIdx = r

		assert bestIdx != -1

		out.append(remainingPoints.pop(bestIdx))

	return out


if __name__ == '__main__':
	seed(0)
	gN = 120
	#gPoints = [ [random(), random()] for _ in range(gN)]
	gPoints = [ [gauss(0, 1), gauss(0, 1)] for _ in range(gN)]

	import cProfile
	import re
	points = initialGuess(gPoints)
	assert len(points) == len(gPoints)
	msr = multiLevelOptimization(points, iters=50)
	#cProfile.run("multiLevelOptimization(gPoints)")

	route = msr.getRoute()

	import matplotlib.pyplot as plt
	X = []
	Y = []
	for i in range(len(route)):
		X.append(route[i][0])
		Y.append(route[i][1])
	#X.append(0.0)
	#Y.append(0.0)
	X.append(X[0])
	Y.append(Y[0])
	plt.plot(X, Y, '-o')
	plt.ylabel('some numbers')
	plt.show()

	
	

