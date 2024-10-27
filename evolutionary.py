#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random
import csv
import getopt, sys

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 50 indexes of cities shuffled around
toolbox.register("individual", tools.initIterate, creator.Individual, lambda: random.sample(list(range(cityN)), cityN))

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def evalCost(individual):
    sumCost = 0.0

    for i in range(len(individual)):
        try:
            if i == 0:
                sumCost += float(costM[individual[-1]][individual[i]])

            else:
                sumCost += float(costM[individual[i-1]][individual[i]])
        except:
            return float('inf'),
    return sumCost,

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evalCost)

# register the crossover operator
toolbox.register("mate", tools.cxPartialyMatched)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.02)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize = 4)

#----------

def main():
    #random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    argList = sys.argv[1:]
    options = "hf:n:c:"
    csvName = "timetrain.csv"
    popN = 100
    global cityN
    cityN = 30
    try:
        
        arguments, values = getopt.getopt(argList, options, "")
        for arg, value in arguments:
            if arg == "-h":
                print("-f  .csv file with cities and costs Default: timetrain.csv\n-n  Population size Default: 100 \n-c Nunber of cities Default: 30")
                exit()
            elif arg == "-f":
                csvName = value
            elif arg == "-n":
                popN = int(value)
            elif arg == "-c":
                cityN = int(value)
        
    except getopt.error as err:
        print(str(err))

    with open(csvName, "r") as fp:
        reader = csv.reader(fp)
        #Get first row
        cities = next(reader)[1:cityN +1]
        #Get Cost matrix
        global costM 
        costM = []
        for i, row in enumerate(reader):
            if i >= cityN:
                break
            else:
                costM.append(row[1:cityN +1])


    pop = toolbox.population(n=popN)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.6, 0.4
    
    print("Start of evolution")
    
    # Evaluate the entire population
    e = 0
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        e += 1
    
    #print("  Evaluated %i individuals" % len(pop))
    #print("  Evaluated %i total individuals" % e)

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    # Begin the evolution
    while  e < 10000:
        # A new generation
        g = g + 1
    #    print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        pop = toolbox.select(pop, popN)
        offspring = toolbox.select(pop, (2*popN) // 3 )
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            e += 1
            ind.fitness.values = fit

        #print("  Evaluated %i individuals" % len(invalid_ind))
        #print("  Evaluated %i total individuals" % e)
        
        # The population is entirely replaced by the offspring
        pop[popN//3:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
     #  print("  Min %s" % min(fits))
     #  print("  Max %s" % max(fits))
     #  print("  Avg %s" % mean)
     #  print("  Std %s" % std)
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % ([cities[i] for i in best_ind], best_ind.fitness.values))


if __name__ == "__main__":
    main()
