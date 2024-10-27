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
        try:
            if i == 0:
                sumCost += float(costM[individual[1][i]][individual[0][-1]][individual[0][i]])
                sumTime += float(timeM[individual[1][i]][individual[0][-1]][individual[0][i]])

            else:
                sumCost += float(costM[individual[1][i]][individual[0][i-1]][individual[0][i]])
                sumTime += float(timeM[individual[1][i]][individual[0][i-1]][individual[0][i]])
        except:
            return float('inf'), float('inf')

    return sumCost, sumTime

def calculate_hypervolume(pareto_front, max_values):
    """
    Calculate the hypervolume of a Pareto front.

    Parameters:
    - pareto_front: A list or array of points on the Pareto front (each point should be a tuple of two values).
    - max_values: A tuple containing the maximum values for the two functions to be minimized.

    Returns:
    - Hypervolume value.
    """
    # Ensure pareto_front is a numpy array for easier manipulation
    pareto_front = np.array(pareto_front)

    # Initialize hypervolume
    hypervolume = 0.0

    # Sort Pareto front points based on the first objective (minimization)
    sorted_front = pareto_front[np.argsort(pareto_front[:, 0])]

    # Compute the hypervolume using the "step" method
    previous_x = 0
    for point in sorted_front:
        x, y = point
        # Calculate width and height of the rectangle
        width = x - previous_x
        height = max_values[1] - y
        
        # Update hypervolume
        hypervolume += width * height
        
        # Update previous_x to current x
        previous_x = x

    return hypervolume

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

    argList = sys.argv[1:]
    options = "hf:n:c:"
    csvOpt = ""
    popN = 100
    global cityN
    cityN = 30
    limits = [0.0, 0.0]

    #Gestão de argumentos de entrada
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

    #Matris de custo
    #   Lista de 3 matrizes cityN*cityN correspondentes aos ficheiros costtrain.csv, costplane.csv e costbus.csv
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
                    for l in row[1:cityN+1]:
                        try:
                            limits[0] = max(float(l), limits[0])
                        except: continue
    
    #Igual ao de cima mas para tempo
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
                    for l in row[1:cityN+1]:
                        try:
                            limits[1] = max(float(l), limits[1])
                        except: continue
    
    #Limits para calculo de Hypervolume
    limits[0] = limits[0]*cityN
    limits[1] = limits[1]*cityN
    print(limits)

    pop = toolbox.population(n=popN)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB1, MUTPB2 = 0.5, 0.2, 0.2
    
    print("Start of evolution")
    
    # Evaluate the entire population
    e = 0
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        e += 1


    # Variable keeping track of the number of generations
    g = 0

    #Create ParetoFront and update it with population
    pareto = tools.ParetoFront()
    pareto.update(pop)

    #  Plot the initial Pareto front
    def plot_pareto_front(pareto_front, non_dominated):
        plt.figure(figsize=(8, 6))
        plt.scatter([ind.fitness.values[0] for ind in pareto_front],
                    [ind.fitness.values[1] for ind in pareto_front],
                    c='blue', label='Pareto Front')
        plt.scatter([ind.fitness.values[0] for ind in non_dominated],
                    [ind.fitness.values[1] for ind in non_dominated],
                    c='red', label='Current pop Pareto Front')
        plt.title('Pareto Front')
        plt.xlabel('Cost (f1)')
        plt.ylabel('Time (f2)')
        plt.legend()
        plt.grid()
        plt.show(block=False)
        plt.pause(1)


    # Calculate hypervolume for the current generation
    non_dominated = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    # Plot initial Pareto front
    plot_pareto_front(pareto, non_dominated)
    plt.close('all')

    # Begin the evolution
    while  e < 10000:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        pop = toolbox.select(pop, popN )
        offspring = toolbox.select(pop, popN // 2 )

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
        pop[popN//2:] = offspring
        
        #pareto front
        pareto.update(pop)

        
        # Calculate hypervolume for the current generation
        non_dominated = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        hv = calculate_hypervolume([ind.fitness.values for ind in non_dominated], limits)
        hv_pareto = calculate_hypervolume([ind.fitness.values for ind in pareto], limits)
        if g%10 == 0:
            # Plot initial Pareto front
            plot_pareto_front(pareto, non_dominated)
            plt.close('all')
        print(f"Generation {g + 1}: Hypervolume = {hv}")
        print(f"Generation {g + 1}: Hypervolume Hall of Fame= {hv_pareto}")

        

    print("-- End of (successful) evolution --")
    
    
    # Calculate hypervolume for the current generation
    non_dominated = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    hv = calculate_hypervolume([ind.fitness.values for ind in non_dominated], limits)
    hv_pareto = calculate_hypervolume([ind.fitness.values for ind in pareto], limits)
    # Plot initial Pareto front
    plot_pareto_front(pareto, non_dominated)
    input()

    for best_ind in pareto: 
        print("Best individual is %s, %s \n\n\n" % ([(cities[i] + "-" + ["train", "plane", "bus"][j]) for i,j in zip(best_ind[0], best_ind[1])], best_ind.fitness.values))

if __name__ == "__main__":
    main()
