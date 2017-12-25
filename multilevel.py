"""
This approach groups points into sub-routes.
"""

from random import randint, random, seed
from copy import copy
from math import sqrt, exp, fabs

from subRoute import *
from metaSubRoute import *


def calcDist(p1, p2):
	return sqrt(sum((p1[i]-p2[i])*(p1[i]-p2[i]) for i in range(len(p1))))


def combineTwoSubRoutes(subroute1, subroute2, case):
	"""
	case 0: Endpoint 0 of subroute 1 to endpoint 0 of subroute 2
	case 1: Endpoint 0 of subroute 1 to endpoint 1 of subroute 2
	case 2: Endpoint 1 of subroute 1 to endpoint 0 of subroute 2
	case 3: Endpoint 1 of subroute 1 to endpoint 1 of subroute 2
	"""

	endpoint1 = None
	endpoint2 = None

	if case==0 or case==1:
		endpoint1 = subroute1.endPoints[0]
		subroute1.points.append(subroute1.endPoints[1])
	else:
		endpoint1 = subroute1.endPoints[1]
		subroute1.points.insert(0, subroute1.endPoints[0])

	if case==0 or case==2:
		endpoint2 = subroute2.endPoints[0]
		subroute2.points.append(subroute2.endPoints[1])
	else:
		endpoint2 = subroute2.endPoints[1]
		subroute2.points.insert(0, subroute2.endPoints[0])

	if case==2 or case==3:
		subroute1.points.reverse()
	if case==1 or case==3:
		subroute2.points.reverse()

	points = subroute1.points + subroute2.points

	subroute = SubRoute(points, endpoint1, endpoint2, dist=subroute1.dist+subroute2.dist + calcDist(endpoint1, endpoint2))

	return subroute


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

