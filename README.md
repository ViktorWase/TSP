# TSP
A multilevel algorithm for symmetric Traveling Salesman Problem. The algorithm divides the cities into chunks based on the distance between them (smaller distance => more likely to be in the same chunk). It then optimizes the inside of the chunks using global optimization algorithms, and then tries to find a good ordering of the chunks. Note that ordering the chunks is basically the TSP problem, but with a reduced number of cities.

The chunks are recursively divided into smaller parts, until they contain only one city at which point they start growing again.

![Example solution](https://github.com/ViktorWase/Portfolio/blob/master/images/tsp.gif)
