#include<stdio.h>
#include <stdlib.h>
#include <math.h>

#define real double

#define DIMS 2

real rand_real(){
	real out = (rand()%10000)/10000.0;
	return out;
}

real dist(real* p1, real* p2){
	return sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]));
}

real calc_cost(const int n, real** cities, int* seq){
	real cost = 0.0;

	for(int i=0; i<n-1; i++){
		cost += dist(cities[seq[i]], cities[seq[i+1]]);
	}

	cost += dist(cities[seq[n-1]], cities[seq[0]]);

	return cost;
}

int main()
{
    printf("Starting.\n");

    const int n = 10;
    const int dim = DIMS;

    int* seq = (int*) malloc(n*sizeof(int));

    //Create random cities
    real** cities = (real**) malloc(n*sizeof(real*));
    for(int i=0; i<n; i++){
    	cities[i] = (real*) malloc(dim*sizeof(real));
    	for(int j=0; j<dim; j++){
    		cities[i][j] = rand_real();
    	} 
    	printf("City: %f %f\n", cities[i][0], cities[i][1]);

    	seq[i] = i;
    }

    printf("Total cost: %f\n", calc_cost(n, cities, seq));

    //Free cities
    for(int i=0; i<n; i++){
    	free(cities[i]);
    }
    free(cities);

    free(seq);

    printf("Done!\n");
    return 0;
}