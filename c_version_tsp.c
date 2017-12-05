#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#define real double
#define REAL_INF 1.0e100

#define DIMS 2

void sanity_check(int* seq, const int n){
	for(int i=0; i<n; i++){
		bool has_found_it = false;
		for(int j=0; j<n; j++){
			if(seq[j] == i){
				has_found_it = true;
				break;
			}
		}
		assert(has_found_it);
	}
}

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


real find_best_point_move(const int pnt_idx, const int n, real** cities, int* seq){
	/*
		Go through all lines and check how much better/worse the tour would be if 
		the city (pnt_idx) was removed from its current place and inserted in between
		the points in the line.
	*/

	// Find the index of the pnt_idx in the list seq
	int idx_in_seq = -1;
	for(int i=0; i<n; i++){ //TODO: This feel like overkill.
		if (seq[i] == pnt_idx){
			idx_in_seq = i;
			break;
		}
	}
	assert(idx_in_seq != -1);

	real best_val_yet = -REAL_INF;
	int best_idx_yet = -1;

	int city_before_p, city_after_p;

	// Go thru all lines and see how much of an improvement it would be.
	for(int line_idx=0; line_idx<n; line_idx++){

		// Ignore the lines that the point belong to.
		if(line_idx!=idx_in_seq && ((line_idx+1)%n)!=idx_in_seq){
			city_before_p = idx_in_seq > 0 ? seq[idx_in_seq-1] : seq[n-1];
			city_after_p = idx_in_seq<n-1 ? seq[idx_in_seq+1] : seq[0];

			real gain = dist(cities[city_before_p], cities[pnt_idx]) + dist(cities[city_after_p], cities[pnt_idx]) - dist(cities[city_before_p], cities[city_after_p]);
			assert(gain >= 0.0);

			int city1_in_line = seq[line_idx];
			int city2_in_line = line_idx < n-1 ? seq[line_idx+1] : seq[0];

			real loss = dist(cities[city1_in_line], cities[pnt_idx]) + dist(cities[city2_in_line], cities[pnt_idx]) - dist(cities[city1_in_line], cities[city2_in_line]);
			assert(loss >= 0.0);

			real val = gain-loss;

			if (val > best_val_yet){
				best_val_yet = val;
				best_idx_yet = line_idx;
			}
		}
	}
	assert(best_idx_yet != -1);

	real out = 0.0;
	// Move the point if the best line is an improvement to the current seq
	if (best_val_yet > 0.0){
		/*
		for (int i = 0; i < n; ++i)
		{
			printf("%d ", seq[i]);
		}
		printf(" \n");
		*/

		int best_line_idx = (best_idx_yet )%n; //Converting line index to point index.
		//printf("best idx: %d\n", best_idx_yet);

		if( idx_in_seq < best_line_idx ){
			for(int i=idx_in_seq; i<best_line_idx; i++){
				seq[i] = seq[(i+1)%n];
			}
			seq[best_line_idx] = pnt_idx;
			//printf("case 1\n");
			
		}
		else{
			for(int i=idx_in_seq; i>best_line_idx; i--){
				seq[i] = seq[(i-1+n)%n];
			}
			seq[(best_line_idx+1)%n] = pnt_idx;
			/*
			printf("case 2\n");
			for (int i = 0; i < n; ++i)
			{
				printf("%d ", seq[i]);
			}
			printf(" \n");
			printf("pnt_idx: %d idx_in_seq: %d\n", pnt_idx, idx_in_seq);
			*/
		}

		out = best_val_yet;
		//printf("out: %f\n", out);
	}

	sanity_check(seq, n);

	//printf(" \n\n");

	return out;
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
		//if( prev_part_cost != new_part_cost){
		//	printf("I don't think this can happen, can it? %f\n", prev_part_cost - new_part_cost);
		//}
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

real cross_untie_and_point_move_heuristic(const int n, real** cities, int *seq){
	real tour_cost = calc_cost(n, cities, seq);
	int itr=0;
	bool should_keep_going = true;
	real improvement;

	real val_prev, val_new;

	while(should_keep_going){
		bool has_made_improvement_this_itr = false;


		for(int pnt_idx=0; pnt_idx<n; pnt_idx++){
			val_prev = calc_cost(n, cities, seq);
			improvement = find_best_point_move(pnt_idx, n, cities, seq);

			if(improvement!=0.0){
				has_made_improvement_this_itr = true;
				tour_cost -= improvement;

				val_new = calc_cost(n, cities, seq);

				//printf("improvement %f,  prev-new: %f, diff: %f\n", improvement, val_prev-val_new, fabs((val_prev-val_new) - improvement ));

				assert( fabs((val_prev-val_new) - improvement ) < 1.0e-8);
			}
		}

		
		val_prev = calc_cost(n, cities, seq);
	    improvement = find_and_untie_crosses(n, cities, seq);
	    if(improvement!=0.0){
			has_made_improvement_this_itr = true;
			tour_cost -= improvement;

			val_new = calc_cost(n, cities, seq);

			assert( fabs((val_prev-val_new) - improvement ) < 1.0e-8);
		}
		

		if(!has_made_improvement_this_itr){
			should_keep_going = false;
		}
		printf("Tour cost: %f, itr: %d\n", tour_cost, itr );
		itr++;
	}

	return tour_cost;
}

int main()
{
    printf("Starting.\n");

    const int n = 1500;
    const int dim = DIMS;

    srand(0);

    int* seq = (int*) malloc(n*sizeof(int));

    //Create random cities and a random tour.
    real** cities = (real**) malloc(n*sizeof(real*));
    for(int i=0; i<n; i++){
    	cities[i] = (real*) malloc(dim*sizeof(real));
    	for(int j=0; j<dim; j++){
    		cities[i][j] = rand_real();
    	} 
    	//printf("City: %f %f\n", cities[i][0], cities[i][1]);

    	seq[i] = i;
    }

    printf("Total end cost: %f\n", cross_untie_and_point_move_heuristic(n, cities, seq));

    //Free cities
    for(int i=0; i<n; i++){
    	free(cities[i]);
    }
    free(cities);

    free(seq);

    printf("Done!\n");
    return 0;
}