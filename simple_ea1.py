#
# THE LOWER THE FITNESS, THE BETTER
#

from random import choice, shuffle, random
import matplotlib.pyplot as plt
import seaborn as sns


### EVOLUTIONARY ALGORITHM ###

def evolve():
    population = create_pop()
    fitness_population = evaluate_pop(population)
    worst_fit = 0
    worst_gen = 0
    finalBest_gen = 0
    finalBest_fit = 1
    finalBest_ind = []
    worst_ind = []
    x_axis = []
    y_axis = []
    for gen in range(NUMBER_GENERATION):
        if gen == 9990:
            pass
        mating_pool = select_pop(population, fitness_population)
        offspring_population = crossover_pop(mating_pool)
        population = mutate_pop(offspring_population)
        fitness_population = evaluate_pop(population)
        best_ind, best_fit = best_pop(population, fitness_population)
        """print("#%3d" % gen, "fit:%.5f" % best_fit)
        print('\n'.join(' |'.join(str(x) for x in row) for row in best_ind))
        print("_______________________")"""
        x_axis.append(gen)
        y_axis.append(best_fit)

        if best_fit > worst_fit:
            worst_fit = best_fit
            worst_gen = gen
            worst_ind = best_ind
        if best_fit < finalBest_fit:
            finalBest_gen = gen
            finalBest_fit = best_fit
            finalBest_ind = best_ind

    sns.regplot(x=x_axis, y=y_axis, line_kws={'color':'red'})
    # plt.plot(x_axis, y_axis)
    print('\n'.join(' |'.join(str(x) for x in row) for row in best_ind))
    title = "Fit over Gen. Generations: %s, Population: %f" %(NUMBER_GENERATION, int(POPULATION_SIZE))
    plt.title(title)
    plt.xlabel('Gen')
    plt.ylabel('Fit')
    plt.show()


    print("Best plot")
    print('\n'.join(' |'.join(str(x) for x in row) for row in finalBest_ind))
    print(finalBest_gen)
    print(finalBest_fit)

    print("Worst plot")
    print('\n'.join(' |'.join(str(x) for x in row) for row in worst_ind))
    print(worst_gen)
    print(worst_fit)


### POPULATION-LEVEL OPERATORS ###

def create_pop():
    return [create_ind() for _ in range(POPULATION_SIZE)]


def evaluate_pop(population):
    return [evaluate_ind(individual) for individual in population]


def select_pop(population, fitness_population):
    sorted_population = sorted(zip(population, fitness_population), key=lambda ind_fit: ind_fit[1])
    return [individual for individual, fitness in sorted_population[:int(POPULATION_SIZE * TRUNCATION_RATE)]]


def crossover_pop(population):
    children = []
    for _ in range(int(POPULATION_SIZE/4)):
        child1, child2 = crossover_ind(choice(population), choice(population))
        children.append(child1)
        children.append(child2)
    return population + children
    #return [crossover_ind(choice(population), choice(population)) for _ in range(int(POPULATION_SIZE))]


def mutate_pop(population):
    return [mutate_ind(individual) for individual in population]


def best_pop(population, fitness_population):
    return sorted(zip(population, fitness_population), key=lambda ind_fit: ind_fit[1])[0]


### INDIVIDUAL-LEVEL OPERATORS: REPRESENTATION & PROBLEM SPECIFIC ###
def create_ind():
    creation = []
    for _ in range(INDIVIDUAL_SIZE):
        shuffle(digits)
        creation.append(list(digits))
    return list(creation)


def evaluate_ind(individual):
    fitness = 0
    fitness += unique_finder(individual, "Row")  # ideally 0
    fitness += unique_finder(list(zip(*individual)), "Column")  # ideally 0
    fitness += unique_finder(box_finder(individual), "Box")  # ideally 0
    fitness += (CLUES - clue_comparator(individual))  # ideally 0
    fitness = fitness / (72 * 3 + CLUES)
    return fitness
    # return (1 - clue_comparator(individual))/CLUES


def crossover_ind(individual1, individual2):
    #return crossover_joiner([n_pair[1][n_pair[0] % 2] for n_pair in enumerate(zip(crossover_splitter(individual1), crossover_splitter(individual2)))])
    """
    child = []
    options = [individual1, individual2]
    for i in range(0,9):
        child.append(options[(i//3)%2][i])
    return child

    child1 = []
    child2 = []
    for n_pair in enumerate(zip(crossover_splitter(individual1), crossover_splitter(individual2))):
        child1.append(n_pair[1][n_pair[0] % 2])
        child2.append(n_pair[1][(n_pair[0]+1) % 2])
    return crossover_joiner(child1), crossover_joiner(child2)"""


    child1 = []
    child2 = []
    for n_pair in list(zip(individual1, individual2)):
        child1.append(choice(n_pair))
        child2.append(choice(n_pair))
    return child1, child2
    #return [choice(n_pair) for n_pair in list(zip(individual1, individual2))]


def mutate_ind(individual):
    digs = digits
    newIndividual = []
    for row in individual:
        if random() < MUTATION_RATE:
            shuffle(digs)
            newIndividual.append(list(digits))
        else:
            newIndividual.append(row)
    return newIndividual


def unique_finder(individual, UnType):
    counter = 0
    try:
        for row in individual:
            counter += (9 - len(set(row)))  # ideally 0
    except TypeError:
        print(individual)
        print(UnType)

    return counter


def box_finder(individual):
    allBox = []
    for i in range(0, 9):
        allBox += individual[i]
    return [box_maker(allBox, i) for i in [0, 3, 6, 27, 30, 33, 54, 57, 60]]


def box_maker(individual, i):
    return individual[i:i + 3] + individual[i + 9:i + 12] + individual[i + 18:i + 21]


def clue_comparator(individual):
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
    grid = open(file, "r")
    targ = grid.read()
    return ''.join(targ.replace('!', '').replace('-', '').split())


def clue_finder(target):
    counter = 0
    for n in target:
        if n != '.':
            counter += 1
    return counter


def crossover_splitter(individual):
    newIndividual = []
    for _, row in enumerate(individual):
        newIndividual.append(row[:3])
        newIndividual.append(row[3:6])
        newIndividual.append(row[6:])
    return newIndividual


def crossover_joiner(individual):
    newIndividual = []
    for i in range(int(len(individual) / 3)):
        newRow = individual[3 * i] + individual[3 * i + 1] + individual[3 * i + 2]
        newIndividual.append(newRow)
    return newIndividual


target = None
CLUES = None
digits = [i for i in range(1, 10)]
INDIVIDUAL_SIZE = 9  # Number of rows

### PARAMERS VALUES ###

NUMBER_GENERATION = 500
POPULATION_SIZE = 1000
TRUNCATION_RATE = 0.5
MUTATION_RATE = 1.0 / INDIVIDUAL_SIZE


### EVOLVE! ###

def main(file):
    global target
    global CLUES
    target = target_setter(file)
    CLUES = clue_finder(target)
    evolve()


main("Grid1.ss")
