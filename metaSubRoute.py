from copy import copy
from math import sqrt, inf
from random import random, randint

from internalOptimization import simulatedAnnealing

def calcDist(p1, p2):
	return sqrt(sum((p1[i]-p2[i])*(p1[i]-p2[i]) for i in range(len(p1))))


class SubRoute():
	"""
	This is a collection of points in a specific order.
	"""
	def __init__(self, points, endpoint1, endpoint2, dist=None, isReversed=True):
		self.endPoints = [endpoint1, endpoint2]
		self.points = list(points)
		self.n = len(points)
		assert self.n>0

		if dist == None:
			self.dist = sum(calcDist(points[i], points[i+1]) for i in range(self.n-1)) + calcDist(endpoint1, points[0]) + calcDist(endpoint2, points[-1])
		else:
			self.dist = dist

		self.idx = -1

		self.isReversed = isReversed

	def getFirstEndPoint(self):
		if self.isReversed:
			return self.endPoints[1]
		else:
			return self.endPoints[0]

	def getSecondEndPoint(self):
		if self.isReversed:
			return self.endPoints[0]
		else:
			return self.endPoints[1]

	def divideInto2SubRoutesRandomly(self):
		"""
		Takes the route and choses a random point. The route is split in 
		two at that point. The two resulting subroutes are returned.
		"""
		# TODO: This function is shit.
		assert self.n > 1
		divisionPoint = randint(0, self.n-1)

		if self.isReversed:
			self.isReversed = False
			self.points.reverse() #TODO: This might be slow

			tmp = self.endPoints[0]
			self.endPoints[0] = self.endPoints[1]
			self.endPoints[1] = tmp

		subroute1 = SubRoute(self.points[0:divisionPoint+1], self.points[0], self.points[divisionPoint])
		subroute2 = SubRoute(self.points[divisionPoint:], self.points[divisionPoint], self.points[-1])

		return (subroute1, subroute2)


	def smoothInternal(self, maxiter=20):
		"""
		Takes the internal route and tries to make it a little bit
		shorter.
		"""
		#assert self.n >= 2 # There's nothing that can be changed otherwise.
		if self.n < 2:
			#print("n is too small. No optimization to be done.")
			return

		(newPoints, newDist) = simulatedAnnealing(copy(self.points), self.endPoints)
		if newDist < self.dist:
			self.dist = newDist
			self.points = newPoints


class MetaSubRoute():
	"""
	A class that contains an assortment of SubRoutes.
	"""
	def __init__(self, subRoutes, connections=None):		
		self.subRoutes = subRoutes
		self.n = len(subRoutes)

		# The list contains the indexes of the elements in the
		# subroutes list. Thus it defines the order of the subroutes.
		if connections==None:
			connections = [i for i in range(len(subRoutes))]
		self.connections = copy(connections)

		self.externalDist = self.calcExternalDist()

	def divideSubroutes(self):
		for i in range(2*self.n):
			assert self.n == len(self.subRoutes)
			# Look at 10 subRoutes and pick the one with the biggest dist.
			# TODO: This 10 is stupid.
			bestYet = -1.0
			bestIdx = -1
			for j in range(10):
				r = randint(0, self.n-1)
				if self.subRoutes[r].dist > bestYet and self.subRoutes[r].n>1:
					bestYet = self.subRoutes[r].dist
					bestIdx = r
			assert bestIdx != -1
			r = bestIdx
			(self.subRoutes[r], newsubroute) = self.subRoutes[r].divideInto2SubRoutesRandomly()

			connectionIdxOfR = -1
			assert r < self.n
			for j in range(len(self.connections)):
				if self.connections[j] == r:
					connectionIdxOfR = j
					break
			assert connectionIdxOfR != -1
			if connectionIdxOfR==self.n-1:
				self.connections.append(self.n)
				self.subRoutes.append(newsubroute)
			else:
				self.connections.insert(connectionIdxOfR+1, self.n)
				self.subRoutes.insert(connectionIdxOfR+1, newsubroute)

			self.n += 1

	def calcExternalDist(self):
		"""
		Calculates the distance between all the subroutes.
		"""
		dist = 0.0
		for i in range(self.n-1):
			p1 = self.subRoutes[self.connections[i]].getSecondEndPoint()
			p2 = self.subRoutes[self.connections[i+1]].getFirstEndPoint()
			dist += calcDist(p1, p2)

		# and the wrap-around-case.
		p1 = self.subRoutes[self.connections[-1]].getSecondEndPoint()
		p2 = self.subRoutes[self.connections[0]].getFirstEndPoint()
		dist += calcDist(p1, p2)

		return dist

	def getTotalDist(self):
		dist = self.externalDist
		for sr in self.subRoutes:
			dist += sr.dist
		return dist


	def optimize(self, maxIter=1000):
		# TODO: Make sure one can REVERSE the subroutes in the optimization.
		def mutate(ind_in, mute_rate):
			ind = copy(ind_in)
			length = len(ind)
			numberOfMutations = randint(1, max(round(length*mute_rate), 1))
			for i in range(numberOfMutations):
				a = randint(0, length - 1)
				b = randint(0, length - 1)
				temp = ind[a]
				ind[a] = ind[b]
				ind[b] = temp
			return ind

		# TODO: We need a proper optimizer here!
		for itr in range(maxIter):
			newConn = mutate(self.connections, random())
			tmp = self.connections
			self.connections = newConn
			newDist = self.calcExternalDist()

			if newDist < self.externalDist:
				self.externalDist = newDist
				print("improvements (external):", newDist)
			else:
				self.connections = tmp


	def combineTwoSubRoutes(self, subroute1, subroute2):
		"""
		Combines 2 subroutes into one.
		"""
		endpoint1 = subroute1.getFirstEndPoint()
		endpoint2 = subroute2.getSecondEndPoint()

		if subroute1.isReversed:
			subroute1.points.insert(0, subroute1.getSecondEndPoint())
			subroute1.points.reverse()
		else:
			subroute1.points.append(subroute1.getSecondEndPoint())

		if subroute2.isReversed:
			subroute2.points.append(subroute2.getFirstEndPoint())
			subroute2.points.reverse()
		else:
			subroute2.points.insert(0, subroute2.getFirstEndPoint())

		points = subroute1.points + subroute2.points

		subroute = SubRoute(points, endpoint1, endpoint2, dist=subroute1.dist+subroute2.dist + calcDist(endpoint1, endpoint2))

		assert len(points) == subroute.n

		return subroute

	def pickWhichSubroutesThatShouldBeCombined(self):
		"""
		Picks a bunch of random subroutes, that are connected, and combines them into one subroute.
		This is repeated a few times.
		"""

		dist = self.externalDist
		iters = int(max((0.5*self.n), 1))

		chosenConnections = [-1 for _ in range(iters)]
		for i in range(iters):
			# Take 5 random connections and pick the shortest one.
			shortestDistYet = inf
			bestIdxYet = -1

			# TODO: 5 is a hard coded number. Do some meta tuning instead.
			for j in range(5):
				r = randint(0, len(self.connections)-1-1)
				p1 = self.subRoutes[self.connections[r]].getSecondEndPoint()
				p2 = self.subRoutes[self.connections[r+1]].getFirstEndPoint()
				dist = calcDist(p1, p2)

				if dist < shortestDistYet:
					shortestDistYet = dist
					bestIdxYet = j

				assert bestIdxYet != -1
				chosenConnections[i] = bestIdxYet

		# Make sure that we don't repeat any combinations (since that would be dumb).
		chosenConnections.sort(reverse=True)
		hasBeenChosen = [False]*(len(self.connections))
		for itr in range(iters):
			if hasBeenChosen[chosenConnections[itr]]:
				pass
			else:
				hasBeenChosen[chosenConnections[itr]] = True
				hasBeenChosen[chosenConnections[itr]+1] = True #TODO: I am not sure regarding this one.
				r = chosenConnections[itr]

				"""
				case = None
				if self.isFirstEndPointInput[r] and self.isFirstEndPointInput[r+1]:
					case = 2

				if self.isFirstEndPointInput[r] and not self.isFirstEndPointInput[r+1]:
					case = 3

				if not self.isFirstEndPointInput[r] and self.isFirstEndPointInput[r+1]:
					case = 0

				if not self.isFirstEndPointInput[r] and not self.isFirstEndPointInput[r+1]:
					case = 1
				assert case != None
				"""
				newSubRoute = self.combineTwoSubRoutes(self.subRoutes[self.connections[r]], self.subRoutes[self.connections[r+1]])

				self.subRoutes[r] = newSubRoute # TODO: Yes?
				#for i in range(r+1, len(self.connections)):
				#	self.connections[i] -= 1
				for i in range(len(self.connections)):
					if self.connections[i]>r:
						self.connections[i] -= 1
				self.connections.pop(r+1) # This might be true?
				self.subRoutes.pop(r+1) # This might be true?
		self.n = len(self.connections)
