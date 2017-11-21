from random import shuffle, randint

def random_division_of_cities(nr_of_cities, nr_of_vans):
	idxs = [i for i in range(nr_of_cities)]
	idxs = shuffle(idxs)

	break_points = [randint(0, nr_of_cities) for _ range(nr_of_vans-1)]

	out = [] * nr_of_vans 

	for idx in idxs:
		



def vrt( cities, nr_of_vans, max_dist_per_van ):
	n = len( cities )


	grade = 0

	division_of_cities = random_division_of_cities()


