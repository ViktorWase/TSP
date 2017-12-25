from copy import copy
from math import sqrt
from random import random, randint

def calcDist(p1, p2):
	return sqrt(sum((p1[i]-p2[i])*(p1[i]-p2[i]) for i in range(len(p1))))

class MetaSubRoute():
	"""
	A class that contains an assortment of SubRoutes.
	"""
	def __init__(self, subRoutes, connections=None, isFirstEndPointInput=None):		
		self.subRoutes = subRoutes
		self.n = len(subRoutes)

		# The list contains the indexes of the elements in the
		# subroutes list. Thus it defines the order of the subroutes.
		if connections==None:
			connections = [i for i in range(len(subRoutes))]
		self.connections = copy(connections)

		if isFirstEndPointInput==None:
			self.isFirstEndPointInput = [True for _ in range(self.n)]
		else:
			self.isFirstEndPointInput = copy(isFirstEndPointInput)

		assert len(self.isFirstEndPointInput) == len(self.connections)

		self.externalDist = self.calcExternalDist()


	def calcExternalDist(self):
		"""
		Calculates the distance between all the subroutes.
		"""
		dist = 0.0
		for i in range(self.n-1):
			p1 = self.subRoutes[self.connections[i]].endPoints[self.isFirstEndPointInput[i]]
			p2 = self.subRoutes[self.connections[i+1]].endPoints[not self.isFirstEndPointInput[i+1]]
			dist += calcDist(p1, p2)

		return dist


	def optimize(self, maxIter=100):

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







