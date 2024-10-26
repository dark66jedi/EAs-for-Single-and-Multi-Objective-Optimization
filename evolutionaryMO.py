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
import numpy as np

import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 50 indexes of cities shuffled around and an index of the transport to use
toolbox.register("individual", tools.initIterate, creator.Individual, lambda: [random.sample(list(range(cityN)), cityN), [random.randint(0,2) for i in range(cityN)]])

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def evalCost(individual):
    sumCost = 0.0
    sumTime = 0.0

    for i in range(len(individual[0])):
        if i == 0:
            sumCost += float(costM[individual[1][i]][individual[0][-1]][individual[0][i]])
            sumTime += float(timeM[individual[1][i]][individual[0][-1]][individual[0][i]])

        else:
            sumCost += float(costM[individual[1][i]][individual[0][i-1]][individual[0][i]])
            sumTime += float(timeM[individual[1][i]][individual[0][i-1]][individual[0][i]])
    return sumCost, sumTime

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evalCost)

# register the crossover operator
toolbox.register("mate", tools.cxPartialyMatched)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutateCities", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("mutateTransport", tools.mutUniformInt, indpb=0.05, low=0, up=2)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selNSGA2)

#----------

def main():
    #random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    argList = sys.argv[1:]
    options = "hf:n:c:"
    csvOpt = ""
    popN = 100
    global cityN
    cityN = 30
    try:
        
        arguments, values = getopt.getopt(argList, options, "")
        for arg, value in arguments:
            if arg == "-h":
                print("-f  Base dir for dataset Default: .\n-n  Population size Default: 100 \n-c Nunber of cities Default: 30")
                exit()
            elif arg == "-f":
                csvOpt = value
            elif arg == "-n":
                popN = int(value)
            elif arg == "-c":
                cityN = int(value)
        
    except getopt.error as err:
        print(str(err))

    global costM 
    costM = [[], [], []]
    for i, j in enumerate(["costtrain.csv", "costplane.csv", "costbus.csv"]):
        csvName =  csvOpt + j
        with open(csvName, "r") as fp:
            reader = csv.reader(fp)
            #Get first row
            cities = next(reader)[1:cityN +1]
            #Get Cost matrix
            for k, row in enumerate(reader):
                if k >= cityN:
                    break
                else:
                    costM[i].append(row[1:cityN +1])

    global timeM 
    timeM = [[], [], []]
    for i, j in enumerate(["timetrain.csv", "timeplane.csv", "timebus.csv"]):
        csvName =  csvOpt + j
        with open(csvName, "r") as fp:
            reader = csv.reader(fp)
            #Get first row
            cities = next(reader)[1:cityN +1]
            #Get Cost matrix
            for k, row in enumerate(reader):
                if k >= cityN:
                    break
                else:
                    timeM[i].append(row[1:cityN +1])

    pop = toolbox.population(n=popN)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB1, MUTPB2 = 0.7, 0.2, 0.2
    
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

    #Create ParetoFront and update it with population
    pareto = tools.ParetoFront()
    pareto.update(pop)

    #  Plot the initial Pareto front
    def plot_pareto_front(pareto_front):
        plt.figure(figsize=(8, 6))
        plt.scatter([ind.fitness.values[0] for ind in pareto_front],
                    [ind.fitness.values[1] for ind in pareto_front],
                    c='blue', label='Pareto Front')
        plt.title('Pareto Front')
        plt.xlabel('Objective 1 (f1)')
        plt.ylabel('Objective 2 (f2)')
        plt.legend()
        plt.grid()
        plt.show(block=False)
        plt.pause(1)


    # Plot initial Pareto front
    plot_pareto_front(pareto)

    # Begin the evolution
    while  e < 10000:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, popN // 3 )
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1[0], child2[0])

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB1:
                toolbox.mutateCities(mutant[0])
                del mutant.fitness.values

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB2:
                toolbox.mutateTransport(mutant[1])
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
        pop[:] = offspring
        
        #pareto front
        pareto.update(pop)
        plot_pareto_front(pareto)
        plt.close('all')
        

    print("-- End of (successful) evolution --")
    
    
    plot_pareto_front(pareto)
    input()

    for best_ind in pareto: 
        print("Best individual is %s, %s" % ([(cities[i] + "-" + ["train", "plane", "bus"][j]) for i,j in zip(best_ind[0], best_ind[1])], best_ind.fitness.values))

if __name__ == "__main__":
    main()
