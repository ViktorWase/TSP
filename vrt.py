from random import shuffle, randint, gauss, seed

def random_division_of_cities(nr_of_cities, nr_of_vans):
	idxs = [i for i in range(nr_of_cities)]
	shuffle(idxs)

	break_points = [randint(0, nr_of_cities) for _ in range(nr_of_vans-1)]
	break_points = sorted(break_points)

	out = [ [] for _ in range( nr_of_vans ) ]

	counter = 0
	van_counter = 0
	for idx in idxs:
		while van_counter != len(break_points) and counter >= break_points[van_counter]:
			van_counter += 1

		out[van_counter].append(idx)
		counter += 1

	return out

		
def vrt( cities, nr_of_vans, max_dist_per_van ):
	n = len( cities )

	grade = 0

	division_of_cities = random_division_of_cities(n, nr_of_vans)

	print(division_of_cities)

if __name__ == '__main__':
	seed(1)
	n = 20
	cities = [ [gauss(0,1), gauss(0,1)] for _ in range(n) ]
	vrt( cities, 2, 100.0 )


