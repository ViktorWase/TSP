from evolutionaryAlgorithm import crossover

import matplotlib.pyplot as plt
from math import sin, cos, pi

cities = []
theta = 0.0
for i in range(8):
	cities.append([cos(theta), sin(theta)])
	theta += pi/4.0

seq1 = [0,1,2,3,4,5,6,7]
seq2 = [0,1,2,3,7,6,5,4]

seq3 = crossover(seq1, seq2)

seqs = [seq1, seq2, seq3]

for i in range(3):
	X = []
	Y = []
	for b in seqs[i]:
		X.append(cities[b][0])
		Y.append(cities[b][1])
	#X.append(0.0)
	#Y.append(0.0)
	X.append(X[0])
	Y.append(Y[0])
	plt.plot(X, Y)

	if i == 0:
		plt.title('First Parent')
	if i==1:
		plt.title('Second Parent')
	if i==2:
		plt.title("Is this a reasonable child?")

	plt.show()