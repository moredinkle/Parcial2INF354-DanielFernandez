import random
import pandas as pd
import numpy

population = []  # list that holds paths
population_size = 10  # max 120 combinations
mutate_prob = 0.1
n_generations = 100
routes_length = [0]*population_size
fitness = [0]*population_size
best_path = 1000
best = []

cities = [0, 1, 2, 3, 4]; cities14 = [1, 2, 3, 4]
df= pd.read_csv(r'C:\Universidad\Septimo\354\grafo.csv', skiprows=0, low_memory=False)
distances = numpy.array(df.values)


def calc_distance(city1, city2):
    return distances[city1][city2]  # ord('A')=65


def create_route():
    shuffled = [0]
    shuffled += random.sample(cities14, len(cities14))
    return shuffled


# calculates length of an route
def calc_route_length():
    for i in range(population_size):
        route_l = 0
        for j in range(0, len(cities)):
            route_l = route_l + calc_distance(population[i][j - 1], population[i][j])
        # route_l = route_l + calc_distance(population[i][len(cities)-1], population[i][1]) calculate distance from last to first
        routes_length[i] = route_l
        fitness[i] = 1 / routes_length[i]



def create_population():
    for i in range(population_size):
        population.append(create_route())



def swap_mutation(ind):
    picks = random.sample(range(len(cities)), 2)
    temp = population[ind][picks[0]]
    population[ind][picks[0]] = population[ind][picks[1]]
    population[ind][picks[1]] = temp


def partially_matched_crossover(ind1, ind2):
    size = len(cities)
    p1, p2 = [0] * size, [0] * size

    # Initialize the position of each indices in the individuals
    for k in range(size):
        p1[ind1[k]] = k
        p2[ind2[k]] = k
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

# Apply crossover between cx points
    for k in range(cxpoint1, cxpoint2):
    # Keep track of the selected values
        temp1 = ind1[k]
        temp2 = ind2[k]
    # Swap the matched value
        ind1[k], ind1[p1[temp2]] = temp2, temp1
        ind2[k], ind2[p2[temp1]] = temp1, temp2
    # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2


def roulette_wheel_selection():
    s = 0
    partial_s = 0
    ind = 0
    for m in range(population_size):
        s = s + fitness[m]
    rand = random.uniform(0, s)
    for m in range(population_size):
        if partial_s < rand:
            partial_s = partial_s + fitness[m]
            ind = ind + 1
    if ind == population_size:  # prevent out of bounds list
        ind = population_size - 1
    return ind


def find_fittest():
    key = 1000
    fittest = 0
    for i in range(population_size):
        if routes_length[i] < key:
            key = routes_length[i]
            fittest = i
    return fittest


create_population()
calc_route_length()
best = [0,0,0,0,0]
rutas = []
for j in range(n_generations):
    for i in range(0, population_size, 2):
       
        parent1 = roulette_wheel_selection()
        parent2 = roulette_wheel_selection()
     
        while True:
            if parent1 == parent2:
                parent2 = roulette_wheel_selection()
            else:
                break
       
        population[i], population[i + 1] = partially_matched_crossover(population[parent1], population[parent2])
       
        calc_route_length()

 
    for i in range(population_size):
        rand = random.uniform(0, 1)
        if rand < mutate_prob:
            swap_mutation(i)

    calc_route_length()

    if routes_length[find_fittest()] < best_path:
        index = find_fittest()
        best_path = routes_length[index]
        
    
    print("Ruta ", j+1, ": ", population[find_fittest()],"Recorrido: ", routes_length[find_fittest()])
    best+=population[find_fittest()]
    rutas.append(routes_length[find_fittest()])
    if routes_length[find_fittest()] == best_path:
        gr = j
    
ind=5*(gr+1)
print("\nMejor ruta: [", best[ind], best[ind+1], best[ind+2], best[ind+3], best[ind+4], "]\nDistancia final:",best_path)
