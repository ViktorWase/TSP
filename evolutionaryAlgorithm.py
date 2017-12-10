from random import shuffle, random, seed, gauss, randint
from copy import copy

from tsp import TSP

def sequence_sanity_check(seq, n):
        """Makes sure that seq is a proper sequence of all cities."""
        assert len(seq) == n
        for i in range(n):
                assert i in seq
        idxs = [i for i in range(n)]
        for x in seq:
                assert x in idxs

def crossover(ind1, ind2):
        """ ind1 will recieve a chunk of ind2 and be adjusted to preserve the 
        chunk.
        """
        giver = ind1
        receiver = copy(ind2)
        lengthSeq = len(giver)

        chunkLength = randint(1,lengthSeq-1)
        chunkPosition =  randint(0,lengthSeq-chunkLength-1)#TODO Off by one? I don't think so, but the line above is.

        chunk = giver[chunkPosition:(chunkPosition + chunkLength)]
        notInChunk = [None] * (lengthSeq-chunkLength)
        outputList = [None] * lengthSeq

        # NOTE: This part is quadratic in time, which is way too slow.
        # It can be reduced to n log n, by sorting copies of the lists
        # and then going through them both together and compiling a 
        # list of all the elements that are in receiver but not in chunk.
        tempCounter = 0
        for i in range(lengthSeq):
                if receiver[i] not in chunk:
                       notInChunk[tempCounter] = receiver[i]
                       tempCounter = tempCounter + 1

        chunkCounter = 0
        notInChunkCounter = 0
        for i in range(lengthSeq):
                if i >= chunkPosition and i < chunkPosition + chunkLength:
                        outputList[i] = chunk[chunkCounter]
                        chunkCounter += 1
                else:
                        outputList[i] = notInChunk[notInChunkCounter]
                        notInChunkCounter += 1

        return(outputList)


def crossover_mod(ind1, ind2):
        """Works exactly the same as the crossover function above, but the
        "chunk" doesn't have to stop when the list stops. It can "wrap around"
        and continue at the beginning of the list. 
        """
        giver = ind1
        receiver = ind2
        lengthSeq = len(giver)

        chunkLength = randint(1,lengthSeq-1)
        chunkPosition =  randint(0,lengthSeq-1)

        chunk = None
        if chunkPosition+chunkLength > lengthSeq:
            chunk = [0.0] * chunkLength
            for i in range(chunkLength):
                chunk[i] = giver[(chunkPosition+i)%lengthSeq]
        else:
            chunk = giver[chunkPosition:(chunkPosition + chunkLength)]
        notInChunk = [0.0] * (lengthSeq-chunkLength)
        outputList = [0.0] * lengthSeq

        tempCounter = 0
        for i in range(lengthSeq):
                if receiver[i] not in chunk:
                       notInChunk[tempCounter] = receiver[i]
                       tempCounter = tempCounter + 1

        if chunkPosition+chunkLength > lengthSeq:
            for i in range(chunkLength):
                outputList[(chunkPosition+i)%lengthSeq] = chunk[i]

            startingPoint = chunkPosition+chunkLength-lengthSeq
            counter = 0
            assert startingPoint >= 0
            assert chunkPosition > startingPoint
            for i in range(startingPoint, chunkPosition):
                outputList[i] = notInChunk[counter]
                counter += 1
        else:
            chunkCounter = 0
            notInChunkCounter = 0
            for i in range(lengthSeq):
                    if i >= chunkPosition and i < chunkPosition + chunkLength:
                            outputList[i] = chunk[chunkCounter]
                            chunkCounter += 1
                    else:
                            outputList[i] = notInChunk[notInChunkCounter]
                            notInChunkCounter += 1

        return(outputList)


def mutate_in_situ(ind, mute_rate=1):
        assert mute_rate >= 0.0

        length = len(ind)
        numberOfMutations = randint(1, max(round(length*mute_rate), 1))

        for i in range(numberOfMutations):
                a = randint(0, length - 1)
                b = randint(0, length - 1)
                temp = ind[a]
                ind[a] = ind[b]
                ind[b] = temp
        
        return(ind)


def mutate(ind, mute_rate=0.1):
        """Mutates the route randomly. 
        It takes 2 random cities and make them switch indexes in
        the route. This process is repeated a random number of times.
        """

        outputList = copy(ind)
        mutate_in_situ(outputList, mute_rate=mute_rate)
        
        return(outputList)

def create_random_individual(n):
        X = [i for i in range(n)]
        shuffle(X)
        return X

def convert_cost_2_fitness(X):
        """Fitness can't be negative, and bigger has to mean better!"""
        x_max = max(X)

        return [x_max-x+1.0e-12 for x in X]

def cumsum_normalized(X):
        total = sum(X)
        out = [0.0]*len(X)

        s = 0.0
        for i in range(len(X)):
                s += X[i]/total
                out[i] = s
        return out

def pick_two_parents(cum_fitness_list):
        # NOTE: Sometimes this one picks the same parent twice. Is that chill?
        r1 = random()
        r2 = random()
        idx1 = -1
        idx2 = -1

        idxs = []
        for _ in range(2):
                r = random()
                for i in range(len(cum_fitness_list)):
                        if r<=cum_fitness_list[i]:
                                idxs.append(i)
                                break
        assert len(idxs) == 2
        return idxs


def ea_tsp(tsp, popsize=50, maxiter=100):
        """Evolutionary Algorithm for Traveling Salesman Problem"""
        nr_of_cities = tsp.n
        dims = tsp.dims

        cities = tsp.cities

        #Create initial population
        population = [create_random_individual(nr_of_cities) for _ in range(popsize)]

        mute_rates = [random() for _ in range(popsize)]

        cost_func = lambda X: tsp.calc_cost(X, should_save=False)

        old_cost = -1

        for itr in range(maxiter):
                cost_list = [cost_func(individual) for individual in population]
                fitness_list = convert_cost_2_fitness(cost_list)
                cumfit_list = cumsum_normalized(fitness_list)

                new_cost = min(cost_list)
                if old_cost != new_cost:
                    old_cost = new_cost
                    print("best yet:", new_cost, "Iter:", itr)
                    print("mean mute:", sum(mute_rates)/(len(mute_rates)))

                mute_rates_next_gen = [-1.0 for _ in range(popsize)]
                next_generation = [[-1]*nr_of_cities for _ in range(popsize)]

                # Create new generation (Not necessary during the last step)
                if itr != maxiter-1:
                        for p in range(popsize):

                                # Pick 2 parents randomly, based on their fitness
                                parent_idxs = pick_two_parents(cumfit_list)

                                # Create and mutate a child.
                                offspring = crossover(population[parent_idxs[0]], population[parent_idxs[1]])

                                # Create new muterate
                                r = random()
                                mute_rate = mute_rates[parent_idxs[0]]*r + mute_rates[parent_idxs[1]]*(1.0-r)
                                mute_rate += gauss(0, 0.05)
                                mute_rate = max(mute_rate, 0)
                                mute_rates_next_gen[p] = mute_rate

                                next_generation[p] = mutate_in_situ(offspring, mute_rate=mute_rate)
                                #next_generation[p] = mutate(offspring)

                                # Check if the output is viable
                                sequence_sanity_check(next_generation[p], nr_of_cities)  #Note: Only for debugging.

                        # Don't forget to copy the best one from the last gen.
                        best_idx = fitness_list.index(max(fitness_list))
                        next_generation[-1] = copy(population[best_idx])
                        mute_rates_next_gen[-1] = mute_rates[best_idx]

                        population = next_generation
                        mute_rates = mute_rates_next_gen


        #Return best one.
        best_idx = fitness_list.index(max(fitness_list))
        return population[best_idx]

if __name__ == '__main__':
        seed(0)
        n = 200
        cities = [[gauss(0,1), gauss(0,1)] for _ in range(n)]

        tsp = TSP(cities)
        tsp.bestYet = ea_tsp(tsp, maxiter=500)
        tsp.calc_cost(copy(tsp.bestYet), should_save=True)
        print("Final cost:", tsp.bestValYet)

        import matplotlib.pyplot as plt
        X = []
        Y = []
        for b in tsp.bestYet:
                X.append(tsp.cities[b][0])
                Y.append(tsp.cities[b][1])
        #X.append(0.0)
        #Y.append(0.0)
        X.append(X[0])
        Y.append(Y[0])
        plt.plot(X, Y)
        plt.ylabel('some numbers')
        plt.show()



