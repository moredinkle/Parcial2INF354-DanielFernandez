# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:41:56 2021

@author: hp
"""


import numpy as np
import array
import random
from deap import base, creator, tools, algorithms
import pandas as pd

def evalTSP(individual):
    distance = distance_map[individual[-1]][individual[0]]
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distance += distance_map[gene1][gene2]
    return distance,


df= pd.read_csv(r'C:\Universidad\Septimo\354\grafo.csv', skiprows=0, low_memory=False)

distance_map = np.array(df.values)
IND_SIZE = 5


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)



toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalTSP)

def main():
    random.seed(169)
    pop = toolbox.population(n=100)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 100, stats=stats, halloffame=hof)
    
    return pop, stats, hof

if __name__ == "__main__":
    print(distance_map)
    pop,stats,hof=main()
    print(hof)
    print(evalTSP(hof[0]))