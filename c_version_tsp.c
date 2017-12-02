#include<stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

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

real untie_cross(int r1, int r2, const int n, real** cities, int* seq){
	/*
		Takes two crossed lines, and uncrosses them.
	*/
	//Make sure that r1 is the small one
	int tmp;
	if(r2 < r1){
		tmp = r2;
		r2 = r1;
		r1 = tmp;
	}

	// Calculate the current cost of the cross
	const real prev_part_cost = dist(cities[seq[r1]], cities[seq[(r1+1)%n]]) + dist(cities[seq[r2]], cities[seq[(r2+1)%n]]);

	// Calculate the cost (of the relevant lines) if the cross would be removed.
	const real new_part_cost = dist(cities[seq[r1]], cities[seq[r2]]) + dist(cities[seq[(r1+1)%n]], cities[seq[(r2+1)%n]]);

	if(new_part_cost < prev_part_cost){
		//tmp = seq[r1+1];
		//seq[r1+1] = seq[r2];
		//seq[r2] = tmp;

		//Flip the order of the sites between r1 and r2
		const int diff = r2 - r1;



		int tmp;
		for(int i=0; i<diff/2; i++){ //TODO: Check that this is correct;
			tmp = seq[r2 - i];
			seq[r2-i] = seq[r1 + i+1];
			seq[r1+i+1] = tmp;
		}

		return prev_part_cost - new_part_cost;
	}
	else{
		printf("I don't think this can happen, can it? %f\n", prev_part_cost - new_part_cost);
		return 0.0;
	}
}

bool do_lines_cross(real** line1, real** line2){
	/*
		Takes 2 lines (defined by their end-points) in 2D and
		checks if they cross.
	*/

	// It doesn't count if they share a point.
	if (line2[0]==line1[0] || line1[0]==line2[1] || line1[1]==line2[0] || line1[1]==line2[1]){
		return false;
	}

	// Check if the 2 lines intersect using linear algebra.
	const real alpha = line1[0][0]-line1[1][0];
	const real beta = -line2[0][0]+line2[1][0];
	const real gamma = line1[0][1]-line1[1][1];
	const real delta = -line2[0][1]+line2[1][1];

	const real det = alpha*delta - gamma*beta;


	if (det != 0.0){ // If they aren't parallel, check where they meet.
		const real epsilon = -line1[1][0]+line2[1][0];
		const real zeta    = -line1[1][1]+line2[1][1];

		const real lambda1 = (delta*epsilon - beta*zeta)/det;
		const real lambda2 = (-gamma*epsilon + alpha*zeta)/det;

		//Do they meet within the end points?
		return lambda1>=0.0 && lambda1<=1.0 && lambda2>=0.0 && lambda2<=1.0;
	}
	else{
		return false;
	}
}

real find_and_untie_crosses(const int n, real** cities, int *seq){
	/*
		This function finds pairs of crossed lines, and then
		switches these lines to make them not cross each other
		any more.

		Complexity: O(n^3)
	*/
	bool found_any_crosses = false;
	int nr_of_crosses = 0;
	real total_improvement = 0.0;

	real** line1 = (real**) malloc(2*sizeof(real*));
	real** line2 = (real**) malloc(2*sizeof(real*));

	line1[0] = (real*) malloc(2*sizeof(real));
	line2[0] = (real*) malloc(2*sizeof(real));
	line1[1] = (real*) malloc(2*sizeof(real));
	line2[1] = (real*) malloc(2*sizeof(real));
	
	for(int i=0; i<n; i++){
		line1[0][0] = cities[seq[i]][0];
		line1[0][1] = cities[seq[i]][1];
		line1[1][0] = cities[seq[(i+1)%n]][0];
		line1[1][1] = cities[seq[(i+1)%n]][1];
		
		for(int j=0; j<n; j++){
			if (abs(i-j) > 1){
				line2[0][0] = cities[seq[j]][0];
				line2[0][1] = cities[seq[j]][1];
				line2[1][0] = cities[seq[(j+1)%n]][0];
				line2[1][1] = cities[seq[(j+1)%n]][1];

				bool has_cross = do_lines_cross(line1, line2);

				if(found_any_crosses==false && has_cross){
					found_any_crosses = true;
				}
				if (has_cross){
					total_improvement += untie_cross( i, j,  n, cities, seq);
					nr_of_crosses += 1;
				}
			}
		}
	}

	free(line1[0]);
	free(line2[0]);
	free(line1[1]);
	free(line2[1]);
	free(line1);
	free(line2);
	return total_improvement;
}

int main()
{
    printf("Starting.\n");

    const int n = 5;
    const int dim = DIMS;

    srand(0);

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

    printf("here: %f \n", find_and_untie_crosses(n, cities, seq));
    printf("Total cost 2: %f\n", calc_cost(n, cities, seq));


    //Free cities
    for(int i=0; i<n; i++){
    	free(cities[i]);
    }
    free(cities);

    free(seq);

    printf("Done!\n");
    return 0;
}