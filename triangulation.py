from copy import deepcopy, copy

class Point():
	def __init__(self, pos, idx, neighs):
		self.pos = pos
		self.idx = idx
		self.neighs = copy(neighs)		

class Triangle():
	def __init__(self, pnt_idxs):
		self.pnt_idxs = copy(pnt_idxs)

class Triangulation():
	def __init__(self, points):
		self.n = len(points)
		self.dims = len(points[0])

		assert self.dims == 2 # I don't plan on supporting anything else.

		self.points = deepcopy(points)

	"""
	def div_and_conq(self):

		class Points():
			#A point in the triangulation. 
			def __init__(self, idx, pos):
				self.idx = idx
				self.pos = pos
				
		class Square():
			#A square with a bunch of points
			def __init__(self, corners, pnt_idxs, parent):
				super(Square, self).__init__()
				self.corners = copy(corners)
				self.pnt_idxs = copy(pnt_idxs)

				self.parent = parent
				self.children = 
	"""

	def create_random_triangulation(self):
		for i in range(n):
			idxs = [i, randint(0, n-1), randint(0, n-1)]
			while idxs[0]==idxs[1] or idxs[1]==idxs[2] or idxs[0]==idxs[2]:
				idxs = [i, randint(0, n-1), randint(0, n-1)]
			triangle = Triangle(idxs)


				
