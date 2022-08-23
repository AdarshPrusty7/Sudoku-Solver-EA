#
# THE LOWER THE FITNESS, THE BETTER
# CODE SHOULD BE EDITED TO REMOVE DATA COLLECTION METHODS IF WANTING TO RUN
#
import csv
from random import choice, shuffle, random
import matplotlib.pyplot as plt
import seaborn as sns


### EVOLUTIONARY ALGORITHM ###

def evolve():
    """
    The 'main' function that runs the algorithm
    The functions are self-explanatory. Visiting them provides more details.
    :return: average fitness float and matplotlib plot for data collection
    """
    population = create_pop()
    fitness_population = evaluate_pop(population)

    averageFit = 0  # Purely for data collection
    x_axis = []  # Graph x-axis
    y_axis = []  # Graph y-axis

    for gen in range(NUMBER_GENERATION):
        # Main algorithm
        mating_pool = select_pop(population, fitness_population)
        offspring_population = crossover_pop(mating_pool)
        population = mutate_pop(offspring_population)
        fitness_population = evaluate_pop(population)
        best_ind, best_fit = best_pop(population, fitness_population)

        # Data collection
        averageFit += best_fit
        x_axis.append(gen)
        y_axis.append(best_fit)

    # Graph plotting + Data
    averageFit = averageFit / 200  # average fitness over 200 generations
    sns.regplot(x=x_axis, y=y_axis, line_kws={'color': 'red'})
    title = "Generations: %s, Population: %f, Mutation: %g" % (NUMBER_GENERATION, int(POPULATION_SIZE), MUTATION_RATE)
    plt.title(title)
    plt.xlabel('Gen')
    plt.ylabel('Fit')

    return averageFit, plt


### POPULATION-LEVEL OPERATORS ###

def create_pop():
    """
    Creates a population
    :return: a list of solutions
    """
    return [create_ind() for _ in range(POPULATION_SIZE)]


def evaluate_pop(population: list):
    """
    Fitness function
    :param population: a list of solutions
    :return: a list of floats that correlate to each element in population
    """
    return [evaluate_ind(individual) for individual in population]


def select_pop(population: list, fitness_population: list):
    """
    Tournament Selection function
    :param population: a list of solutions
    :param fitness_population: a list of floats that correlate to each element in population
    :return: a list of 20% of population's size containing generally fit results
    """
    newPop = []
    sorted_population = list(zip(population, fitness_population))
    #
    for i in range(int(POPULATION_SIZE / 5)):  # 5 because we're selecting 2 from every 5
        parents = [(sorted_population.pop()) for _ in range(5)]  # Select 5
        parents = sorted(parents, key=lambda x: x[1])  # Sort it and choose 2 best solutions
        newPop.append(parents[0][0])
        newPop.append(parents[1][0])
    return newPop


def crossover_pop(population: list):
    """
    Creates two children per set of parents
    :param population: list of solutions
    :return: list of parents + newly created children
    """
    children = []
    for _ in range(int(POPULATION_SIZE * 6 / 20)):  # 6/20=0.6*0.5 a) 0.6 because 0.4 of parents already exist and b) 0.5 because 2 children are created
        child1, child2 = crossover_ind(choice(population), choice(population))  # Creates 2 children
        children.append(child1)
        children.append(child2)
    return population + children  # Final size should be POPULATION_SIZE


def mutate_pop(population: list):
    """
    Mutates population
    :param population: list of population
    :return: list of mutated population
    """
    return [mutate_ind(individual) for individual in population]


def best_pop(population: list, fitness_population: list):
    """
    Sorts population by best fitness
    :param population: list of population
    :param fitness_population: list of floats that correlate to each element in population
    :return: a collection of population and fitness
    """
    return sorted(zip(population, fitness_population), key=lambda ind_fit: ind_fit[1])[0]


### INDIVIDUAL-LEVEL OPERATORS: REPRESENTATION & PROBLEM SPECIFIC ###
def create_ind():
    """
    Creates each member of the population
    :return: a list of rows (member of population)
    """
    creation = []
    for _ in range(INDIVIDUAL_SIZE):
        shuffle(digits)  # digits is all digits 1-9 so this keeps every row unique
        creation.append(list(digits))
    return list(creation)


def evaluate_ind(individual: list):
    """
    Evaluates each member
    :param individual: a list of rows (member of population)
    :return: a float
    """
    fitness = 0
    fitness += unique_finder(individual, "Row")  # ideally 0
    fitness += unique_finder(list(zip(*individual)), "Column")  # ideally 0
    fitness += unique_finder(box_finder(individual), "Box")  # ideally 0
    fitness += (CLUES - clue_comparator(individual))  # ideally 0
    fitness = fitness / (72 * 3 + CLUES)  # maximum is 72 * 3 + CLUES so always between 0 and 1
    return fitness


def crossover_ind(individual1: list, individual2: list):
    """
    Crossover for two parents
    :param individual1: list (Parent 1)
    :param individual2: list (Parent 2)
    :return: 2 lists of rows (children)
    """
    child1 = []
    child2 = []
    for n_pair in list(zip(individual1, individual2)):
        n_pair = list(n_pair)
        shuffle(n_pair)
        # Tries to maintain diversity by using both rows in either parents
        child1.append(n_pair[0])
        child2.append(n_pair[1])
    return child1, child2


def mutate_ind(individual: list):
    """
    Mutates individual randomly
    :param individual:
    :return:
    """
    digs = digits
    newIndividual = []
    for row in individual:
        if random() < MUTATION_RATE:  # Random mutation
            shuffle(digs)
            newIndividual.append(list(digits))
        else:
            newIndividual.append(row)  # If not mutated, add it back
    return newIndividual


def unique_finder(individual: list, UnType: str):
    """
    Finds number of unique digits in UnType
    :param individual: list of rows (member of population)
    :param UnType: string denoting what section of grid is being evaluated (for debugging purposes)
    :return: int, fitness
    """
    counter = 0
    try:
        for row in individual:
            counter += (9 - len(set(row)))  # ideally 0. set removes duplicate elements.
    except TypeError:
        print(individual)
        print(UnType)
    return counter


def box_finder(individual: list):
    """
    Makes individual suitable for unique_finder
    :param individual: a list of rows (member of population)
    :return:a list of 'rows' (really, just a 3x3 box)
    """
    allBox = []
    for i in range(0, 9):
        allBox += individual[i]
    return [box_maker(allBox, i) for i in [0, 3, 6, 27, 30, 33, 54, 57, 60]]


def box_maker(individual: list, i: int):
    """
    Separates grid into 3x3 boxes and returns them as row
    :param individual: list of rows
    :param i: index position in row
    :return: list of 'rows'
    """
    return individual[i:i + 3] + individual[i + 9:i + 12] + individual[i + 18:i + 21]


def clue_comparator(individual: list):
    """
    Checks how close each individual is to the target
    :param individual: list of rows
    :return: int, closeness of clues
    """
    counter = 0
    flatAll = []
    for i in range(0, 9):
        flatAll += individual[i]
    for n in enumerate(target):
        if n[1] != '.':
            if int(n[1]) == flatAll[n[0]]:  # if they do match add one to fitness
                counter += 1
    return counter


def target_setter(file):
    """
    Formats the .ss file for this program
    :param file: grid file
    :return: string target
    """
    grid = open(file, "r")
    targ = grid.read()
    return ''.join(targ.replace('!', '').replace('-', '').split())


def clue_finder(target):
    """
    Finds out how many clues there are
    :param target: list of ints
    :return: int, number of clues
    """
    counter = 0
    for n in target:
        if n != '.':
            counter += 1
    return counter


# Sets up project values
target = None
CLUES = None
digits = [i for i in range(1, 10)]
INDIVIDUAL_SIZE = 9  # Number of rows

### PARAMERS VALUES ###

NUMBER_GENERATION = 200
POPULATION_SIZE = int(10000)
TRUNCATION_RATE = 0.4
MUTATION_RATE = 1.2 / INDIVIDUAL_SIZE


### EVOLVE! ###

# This file is used to run evolve AND collect data.
def main(file):
    global target
    global CLUES
    target = target_setter(file)
    CLUES = clue_finder(target)
    average, pl = evolve()
    return average, pl

# This iteration is PURELY for data collection
NUMBER_GENERATION = 200
for i in range(3):
    file = ""
    writ = ""
    name = ""
    if i == 0:
        file = "Grid1.ss"
        writ = "grid1.csv"
        name = "grid1"
    elif i == 1:
        file = "Grid2.ss"
        writ = "grid2.csv"
        name = "grid2"
    elif i == 2:
        file = "Grid3.ss"
        writ = "grid3.csv"
        name = "grid3"
    POPULATION_SIZE = 10
    for i in range(5):
        average, pl = main(file)
        f = open(writ, "a")
        writer = csv.writer(f)
        writer.writerow([POPULATION_SIZE, average])
        f.close()
        pl.savefig("./graphs/%s_%f_%g.png" % (name, POPULATION_SIZE, i))
        pl.clf()

    POPULATION_SIZE = 100
    for i in range(5):
        average, pl = main(file)
        f = open(writ, "a")
        writer = csv.writer(f)
        writer.writerow([POPULATION_SIZE, average])
        f.close()
        pl.savefig("./graphs/%s_%f_%g.png" % (name, POPULATION_SIZE, i))
        pl.clf()

    POPULATION_SIZE = 1000
    for i in range(5):
        average, pl = main(file)
        f = open(writ, "a")
        writer = csv.writer(f)
        writer.writerow([POPULATION_SIZE, average])
        f.close()
        pl.savefig("./graphs/%s_%f_%g.png" % (name, POPULATION_SIZE, i))
        pl.clf()

    POPULATION_SIZE = 10000
    for i in range(5):
        average, pl = main(file)
        f = open(writ, "a")
        writer = csv.writer(f)
        writer.writerow([POPULATION_SIZE, average])
        f.close()
        pl.savefig("./graphs/%s_%f_%g.png" % (name, POPULATION_SIZE, i))
        pl.clf()
