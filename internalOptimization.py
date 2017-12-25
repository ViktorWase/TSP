from math import sqrt, exp, fabs
from random import random, gauss, randint
from copy import copy

def calcDist(p1, p2):
	return sqrt(sum((p1[i]-p2[i])*(p1[i]-p2[i]) for i in range(len(p1))))

def mutate_using_2_opts(route, endpoints):
	n = len(route)
	r1 = randint(0, n-1-1)

	r2 = randint(0, n-1-1)

	counter = 0
	while r2 == r1 or abs(r1-r2)==1:
		r2 = randint(0, n-1-1)

		counter+=1
		if counter > 1000000:
			assert False

	#Make sure that r1 is the small one
	if r2 < r1:
		tmp = r2
		r2 = r1
		r1 = tmp
	prev_part_cost = calcDist(route[r1], route[r1+1]) + calcDist(route[r2], route[r2+1])
	new_part_cost = calcDist(route[r1], route[r2]) + calcDist(route[r1+1], route[r2+1])

	tmp = route[r1+1]
	route[r1+1] = route[r2]
	route[r2] = tmp

	#Flip the order of the sites between r1 and r2
	mid = copy(route[(r1+2):r2])
	mid.reverse()
	counter = 0
	for i in range(r1+2, r2):
		route[i] = mid[counter]
		counter += 1

	return (route, -(new_part_cost-prev_part_cost))


def movePoint(route, endPoints):
	"""
	Takes a random point and moves it to a random position in the route.
	"""
	n = len(route)
	r1 = randint(0, len(route)-1) # Chose a random point
	r2 = randint(0, len(route)) #Chose a random line (note that there is one more line than point).
	while r2-1 == r1 or r2==r1:
		r2 = randint(0, len(route))

	city_before_p = route[r1-1] if r1>0 else endPoints[0]
	city_after_p = route[r1+1] if r1<n-1 else endPoints[1]

	chosenCity = route[r1]

	gain = calcDist(city_before_p, chosenCity) + calcDist(city_after_p, chosenCity) - calcDist(city_before_p, city_after_p)
	assert gain >= 0.0

	city1_in_line = route[r2-1] if r2>0 else endPoints[0] 
	city2_in_line = route[r2] if r2<n else endPoints[1]

	loss = calcDist(city1_in_line, route[r1]) + calcDist(city2_in_line, route[r1]) - calcDist(city1_in_line, city2_in_line)
	assert loss >= 0.0


	OLDVAL = sum(calcDist(route[i], route[i+1]) for i in range(len(route)-1)) + calcDist(endPoints[0], route[0]) + calcDist(endPoints[1], route[-1])

	val = route.pop(r1)

	if r2 == n:
		route.append(val)
	elif r2 <= r1:
		route.insert(r2, val)
	else:
		route.insert(r2-1, val)

	NEWVAL = sum(calcDist(route[i], route[i+1]) for i in range(len(route)-1)) + calcDist(endPoints[0], route[0]) + calcDist(endPoints[1], route[-1])

	improvement = gain - loss


	if(fabs( (NEWVAL-OLDVAL) + (improvement) ) >= 1.0e-13 ):
		print("Newval:", NEWVAL, "Oldval:", OLDVAL, "Gain:", gain, "Loss:", loss)
		print("lhs:", NEWVAL-OLDVAL, "rhs", gain-loss)
		print("r1:", r1, "r2:", r2, "n:", n)
	assert fabs( (NEWVAL-OLDVAL) + (improvement) ) < 1.0e-13

	return (route, improvement)


def simulatedAnnealing(route, endPoints, maxIter=100, decay=0.9, startDist=None):
	"""
	Optimization algorithm that is used to improve a route.
	The niegbourhood is a 2-opt swap or a point move.
	"""

	def acceptanceProb(newVal, orgVal, temp):
		if newVal < orgVal:
			return 1.0
		else:
			return exp(-(newVal-orgVal)/temp)

	def getRandNeig(route, endPoints):
		newRoute = list(route) #TODO: This might be super slow
		if len(route) <= 4 or random() < 0.5:
			(newRoute, improvement) = movePoint(newRoute, endPoints)
		else:
			(newRoute, improvement) = mutate_using_2_opts(newRoute, endPoints)
		return (newRoute, improvement)

	def temperature(x):
		return x

	if startDist == None:
		startDist = sum(calcDist(route[i], route[i+1]) for i in range(len(route)-1)) + calcDist(endPoints[0], route[0]) + calcDist(endPoints[1], route[-1])

	currentDist = startDist
	for itr in range(maxIter):
		temp = temperature(float(maxIter-itr)/maxIter)
		(newRoute, improvement) = getRandNeig(route, endPoints)

		ap = acceptanceProb(currentDist-improvement, currentDist, temp)
		if ap >= random():
			currentDist -= improvement
			route = newRoute

			print("New improvement:", currentDist, improvement)

	NEWVAL = sum(calcDist(route[i], route[i+1]) for i in range(len(route)-1)) + calcDist(endPoints[0], route[0]) + calcDist(endPoints[1], route[-1])
	if fabs(NEWVAL-currentDist)>1.0e-8:
		print("newval:", NEWVAL, "currdist:", currentDist, "start:", startDist)
	assert(fabs(NEWVAL-currentDist)<1.0e-8)
	return (route, currentDist)