from multilevel import calcDist

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
		