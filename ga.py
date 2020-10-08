from deap import base
from deap import creator
from deap import tools
import random
import numpy as np
import re

SELECTED_MODEL = None
FIRST_CHROMOSOME = None


def get_toolbox(first_chromosome, selected_model):
    global FIRST_CHROMOSOME
    FIRST_CHROMOSOME = first_chromosome
    global SELECTED_MODEL
    SELECTED_MODEL = selected_model

    chromosome_length = len(first_chromosome)

    chromosomeY = np.array(first_chromosome)
    chromosomeY = chromosomeY.astype('int32')
    chromosomeY = chromosomeY.reshape(-1, 1)
    chromosomeY = chromosomeY.T

    selected_model.predict(chromosomeY)[0]

    # problem constants:
    ONE_MAX_LENGTH = chromosome_length  # length of chromosome to be optimized

    # set the random seed:
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    toolbox = base.Toolbox()

    # create an operator that randomly returns 0 or 1:
    toolbox.register("zeroOrOne", random.randint, 0, 1)

    # define a single objective, maximizing fitness strategy:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # create the Individual class based on list:
    creator.create("Individual", list, fitness=creator.FitnessMax)
    # creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

    # create the individual operator to fill up an Individual instance:
    toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)

    # create the population operator to generate a list of individuals:
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    toolbox.register("evaluate", ruleFitness)

    # Tournament selection with tournament size of 3:
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Single-point crossover:
    toolbox.register("mate", tools.cxOnePoint)

    # Flip-bit mutation:
    # indpb: Independent probability for each attribute to be flipped
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)

    return toolbox


# Prepare function to count the number of bits used in chromosome by each rule variable
def count_var(text_to_search, text_list):
    count = 0
    for col in text_list:
        if re.search(text_to_search, col):
            count +=1
    return count


def ruleFitness(individual):
    # Prepare function to count the number of bits used in chromosome by each rule variable
    # var1 = "Ideal Location of Childcare"
    var1 = "acceptable_distance"
    var2 = "acceptable_fees"
    var3 = "second_language"
    var4 = "dietary_restrictions"
    var5 = "service_type"
    var6 = "childcare_rank"
    var7 = "enrol_reg_diff"
    var8 = "today_reg_diff"
    var9 = "study_level"

    num_var1 = count_var(var1, FIRST_CHROMOSOME.index)
    num_var2 = count_var(var2, FIRST_CHROMOSOME.index)
    num_var3 = count_var(var3, FIRST_CHROMOSOME.index)
    num_var4 = count_var(var4, FIRST_CHROMOSOME.index)
    num_var5 = count_var(var5, FIRST_CHROMOSOME.index)
    num_var6 = count_var(var6, FIRST_CHROMOSOME.index)
    num_var7 = count_var(var7, FIRST_CHROMOSOME.index)
    num_var8 = count_var(var8, FIRST_CHROMOSOME.index)
    num_var9 = count_var(var9, FIRST_CHROMOSOME.index)

    list_num_var = [num_var1, num_var2, num_var3, num_var4, num_var5, num_var6, num_var7, num_var8, num_var9]

    chromosomeY = np.array(individual)
    chromosomeY = chromosomeY.astype('int32')
    chromosomeY = chromosomeY.reshape(-1, 1)
    chromosomeY = chromosomeY.T

    numOnes_per_var = []
    chromo_idx = 0
    len_list_var = len(list_num_var)

    for i in range(len_list_var):
        if list_num_var[i] != 1:

            list_to_num = ''.join(map(str, chromosomeY[0][chromo_idx: chromo_idx + list_num_var[i]]))
            each_char_to_int = [int(c) for c in list_to_num]
            numofones = sum(each_char_to_int)
            numOnes_per_var.append(numofones)
            chromo_idx += list_num_var[i]

        else:
            encoded_num = chromosomeY[0][chromo_idx]
            numOnes_per_var.append(encoded_num)
            chromo_idx += 1

    # CHECK IF CHROMOSOME IS VALID SOLUTION,
    ok = True
    tempsum = 0
    for idx in range(len(numOnes_per_var)):
        if numOnes_per_var[idx] > 1:
            # Not ok if it is not valid
            ok = False
            tempsum -= 10

    if ok:
        # retrieve node ID
        nodeID = SELECTED_MODEL.apply(chromosomeY)
        nodeID = nodeID[0]

        # num of samples in one node / total sample num
        node_samples = SELECTED_MODEL.tree_.n_node_samples[nodeID]
        max_samples = SELECTED_MODEL.tree_.n_node_samples[0]  # ID "0" is root DT node
        samples_percent = node_samples / max_samples * 100

        # get prediction value for current chromosome (value 2.0 resembles "Accept Offer" )
        y_value = SELECTED_MODEL.predict(chromosomeY)[0]

        # final fitness score
        numsum = y_value * 50 + samples_percent

    else:
        numsum = tempsum

    return numsum,


def main(first_chromosome, selected_model):
    # Genetic Algorithm constants:
    ONE_MAX_VAL = 200  # i set this guess
    POPULATION_SIZE = 200
    P_CROSSOVER = 0.9  # probability for crossover
    P_MUTATION = 0.1  # probability for mutating an individual
    MAX_GENERATIONS = 50

    toolbox = get_toolbox(first_chromosome, selected_model)
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    generationCounter = 0

    # calculate fitness tuple for each individual in the population:
    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    # extract fitness values from all individuals in population:
    fitnessValues = [individual.fitness.values[0] for individual in population]

    # initialize statistics accumulators:
    maxFitnessValues = []
    meanFitnessValues = []

    print( fitnessValues )
    print('MAX' + '\n')
    print(max(fitnessValues))

    # main evolutionary loop:
    # stop if max fitness value reached the known max value
    # OR if number of generations exceeded the preset value:
    while max(fitnessValues) < ONE_MAX_VAL and generationCounter < MAX_GENERATIONS:
        # update counter:
        generationCounter = generationCounter + 1

        # apply the selection operator, to select the next generation's individuals:
        offspring = toolbox.select(population, len(population))

        # clone the selected individuals:
        offspring = list(map(toolbox.clone, offspring))

        # apply the crossover operator to pairs of offspring:
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # calculate fitness for the individuals with no previous calculated fitness value:
        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue

        # replace the current population with the offspring:
        population[:] = offspring

        # collect fitnessValues into a list, update statistics and print:
        fitnessValues = [ind.fitness.values[0] for ind in population]

        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        print("- Generation {}: Max Fitness = {}, Avg Fitness = {}".format(generationCounter, maxFitness, meanFitness))

        # find and print best individual:
        best_index = fitnessValues.index(max(fitnessValues))
        print("Best Individual = ", *population[best_index], "\n")

    return population, best_index


def get_top_ga_solutions(pipeline, first_chromosome, selected_model):
    num_sol = 100
    first_chromosome = pipeline.fit_transform(first_chromosome[first_chromosome["cc_action"].notnull()]).iloc[1,:-1]
    pop, best_index = main(first_chromosome, selected_model)
    pop_array = np.array(pop)

    best_chromosomes = pop_array[:num_sol]

    # Retrive OHE
    transf = pipeline.named_steps['column_transform'].transformers_
    ohe = transf[0][1]

    rules_var_values = ohe.inverse_transform(best_chromosomes)

    return rules_var_values

