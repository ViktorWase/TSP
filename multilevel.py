"""
This approach groups points into sub-routes.
"""

from random import randint, random, seed
from copy import copy
from math import sqrt, exp, fabs


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

	return (route, new_part_cost-prev_part_cost)


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


	if(fabs( (NEWVAL-OLDVAL) + (gain-loss) ) >= 1.0e-10 ):
		print("Newval:", NEWVAL, "Oldval:", OLDVAL, "Gain:", gain, "Loss:", loss)
		print("lhs:", NEWVAL-OLDVAL, "rhs", gain-loss)
		print("r1:", r1, "r2:", r2, "n:", n)
	assert fabs( (NEWVAL-OLDVAL) + (gain-loss) ) < 1.0e-10


	return (route, gain-loss)


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
	assert(fabs(NEWVAL-currentDist)<1.0e-8)
	return (route, currentDist)


class SubRoute():
	"""
	This is a collection of points in a specific order.
	"""
	def __init__(self, points, endpoint1, endpoint2, dist=None):
		self.endPoints = [endpoint1, endpoint2]
		self.points = list(points)
		self.n = len(points)
		assert self.n>0

		if dist == None:
			self.dist = sum(self.calcDist(points[i], points[i+1]) for i in range(self.n-1)) + calcDist(endpoint1, points[0]) + calcDist(endpoint2, points[-1])
		else:
			self.dist = dist


	def divideInto2SubRoutesRandomly(self):
		"""
		Takes the route and choses a random point. The route is split in 
		two at that point. The two resulting subroutes are returned.
		"""
		assert self.n > 1
		divisionPoint = randint(0, self.n-1)

		subroute1 = SubRoute(points[0:divisionPoint+1], points[0], points[divisionPoint])
		subroute2 = SubRoute(points[divisionPoint:], points[divisionPoint], points[-1])

		return [subroute1, subroute2]


	def smoothInternal(self, maxiter=20):
		"""
		Takes the internal route and tries to make it a little bit
		shorter.
		"""
		assert self.n > 3 # There's nothing that can be changed otherwise.

		# TODO: Insert some smart combinatorial algorithm here.


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

	biggerSubRoute = combineTwoSubRoutes(subroutes[0], subroutes[1], 0)
	simulatedAnnealing(biggerSubRoute.points, biggerSubRoute.endPoints)






