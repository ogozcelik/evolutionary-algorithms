import cv2
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

# PARAMETERS
num_inds = 75  # 20 default
num_genes = 100  # 50 default
tm_size = 5  # 5 default
frac_elites = 0.05  # 0.2 default
frac_parents = 0.6  # 0.6 default
mutation_prob = 0.5  # 0.2 default
mutation_type = False  # True: unguided, False: guided || False default
num_generations = 16000
generations = 0  # generation counter
source_image = cv2.imread('painting.png')  # read source imginstall
fitness_list = []  # for plot
x_coordinate = []  # x axis
x_coordinate1000 = []
for i in range(1, 10001):
    x_coordinate.append(i)
for i in range(1, 1001):
    x_coordinate1000.append(i)


class population:
    def __init__(self):
        self.population_list = []  # list including individuals
        # self.size = len(population_list)

    def add_individuals(self, individual):
        self.population_list.append(individual)


class individuals:
    def __init__(self, fitness):
        self.genes_list = []  # list including genes actually one chromosome
        self.fitness = fitness

    def sort_radii(self):
        self.genes_list = sorted(self.genes_list, key=lambda x: x.radii, reverse=True)

    def add_genes(self, gene):
        self.genes_list.append(gene)


class genes:
    def __init__(self, x, y, radii, red, green, blue, alpha):
        self.x = x
        self.y = y
        self.radii = radii
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha


def white_img():  # function to create white img
    white = np.zeros([180, 180, 3], dtype=np.uint8)
    white.fill(255)  # or img[:] = 255
    return white


# initialize_pop function properly initializes a population
# using random values and given two parameters and returns
# a class object of population
def initialize_pop(num_individuals, gene_number):
    pop = population()
    for n_i in range(num_individuals):
        ind = individuals(0)  # default fitness is 0
        for n_g in range(gene_number):
            gen = genes(random.randint(0, 179),  # random x
                        random.randint(0, 179),  # random y
                        random.randint(0, 90),  # random radii
                        random.randint(0, 255),  # random red
                        random.randint(0, 255),  # random green
                        random.randint(0, 255),  # random blue
                        random.uniform(0, 1)  # random alpha
                        )
            ind.add_genes(gen)
        pop.add_individuals(ind)
    return pop


# evaluation function takes individual object
# in the function, white image is created and according to algorithm
# the circles are drawn and fitness is calculated. Function returns the fitness value
def evaluation(individual):
    fitness = 0  # default fitness is 0
    w_img = white_img()  # white image
    over_img = np.empty_like(w_img)  # initialize overlay numpy array
    individual.sort_radii()  # sort radii of the individual
    for index in range(num_genes):  # iterate over genes
        over_img[:] = w_img  # copy image to overlay
        # over_img = copy.deepcopy(w_img)
        gene = individual.genes_list[index]  # capture the gene
        cv2.circle(over_img, (gene.x, gene.y), gene.radii, (gene.red, gene.green, gene.blue), -1)  # draw the circle
        # on overlay
        #w_img = (over_img.astype(np.float64) * individual.genes_list[index].alpha + w_img.astype(np.float64) * (
                #1 - individual.genes_list[index].alpha)).astype(np.uint8)  # consider uint8 conversion
        w_img = copy.deepcopy(cv2.addWeighted(over_img,individual.genes_list[index].alpha,w_img,1-individual.genes_list[index].alpha,0))
        # cv2.imshow('img', w_img)
        # cv2.waitKey(0)
    # fitness calculation
    # fitness = 0 - np.sum(np.square(np.subtract(source_image, w_img, dtype='int')))
    global source_image
    source_image = np.int64(source_image)
    w_img = np.int64(w_img)
    for k in range(3):
        individual_single_channel = w_img[:, :, k].flatten()
        source_single_channel = source_image[:, :, k].flatten()
        fitness += -1 * sum((source_single_channel - individual_single_channel) ** 2)
    individual.fitness = fitness  # assign the fitness value to individual
    return fitness


def tournament(pop, tournament_size):  # tournament selection function
    init_random = random.randint(0, len(pop.population_list) - 1)
    best = pop.population_list[init_random]
    index = init_random
    for i in range(tournament_size - 1):
        random_index = random.randint(0, len(pop.population_list) - 1)
        index = random_index
        if pop.population_list[random_index].fitness >= best.fitness:
            best = pop.population_list[random_index]
            index = random_index
    return best, random_index  # returns best and its index in the population_list


def selection(pop, tournament_size):
    elites = []
    next_individuals = []
    # selection of elites to the next generation
    N = frac_elites * num_inds  # number of the elites
    for i in range(int(N)):  # finding elites
        index = 0
        max1 = pop.population_list[index].fitness

        for j in range(1, len(pop.population_list)):
            if pop.population_list[j].fitness > max1:
                max1 = pop.population_list[j].fitness
                index = j
        elites.append(pop.population_list[index])  # adding to the elites
        pop.population_list.pop(index)  # update population
    # DEBUG
    # for i in range(len(elites)):
    # print("ELITE: " + str(elites[i].fitness))
    # for j in range(len(pop.population_list)):
    # print("NOT ELITE: " + str(pop.population_list[j].fitness))
    global generations
    print("GENERATION NUMBER: " + str(generations))
    # tournament selection
    total_turn = len(pop.population_list)  # how many tournements can be done
    for i in range(total_turn):
        best, index = tournament(pop, tournament_size)  # call tournament function
        next_individuals.append(best)  # add winner to the next_individuals
    for h in range(len(pop.population_list)):
        pop.population_list.pop()  # remove all from the population
    # NOW population_list is empty
    for i in range(len(elites)):
        pop.add_individuals(elites[i])  # add elites to the population_list
    return next_individuals  # after this func. ends -> population include elites,


def crossover(pop, tournament_size, gene_number):
    chosen_individuals = selection(pop, tournament_size)
    # parents = []
    parent_number = frac_parents * num_inds
    child = []
    return_list = []
    for i in range(int(parent_number / 2)):
        p1_index = random.randint(0, len(chosen_individuals) - 1)
        parent1 = copy.deepcopy(chosen_individuals.pop(p1_index))
        # parents.append(parent1)  # pop and append the parent

        p2_index = random.randint(0, len(chosen_individuals) - 1)
        parent2 = copy.deepcopy(chosen_individuals.pop(p2_index))
        # parents.append(parent2)  # pop and append the parent

        child1 = copy.deepcopy(parent1)  # child are default as parents
        child2 = copy.deepcopy(parent2)

        for gene_i in range(gene_number):
            choice = random.randint(0, 1)  # choice of randomly selected parents
            if choice == 0:  # if choice is 0
                child1.genes_list[gene_i] = copy.deepcopy(parent1.genes_list[gene_i])
                child2.genes_list[gene_i] = copy.deepcopy(parent2.genes_list[gene_i])
            else:  # if choice is 1
                child1.genes_list[gene_i] = copy.deepcopy(parent2.genes_list[gene_i])
                child2.genes_list[gene_i] = copy.deepcopy(parent1.genes_list[gene_i])
        global generations
        global fitness_list
        child.append(child1)
        generations += 1
        # print("Generation: " + str(generations))
        if generations <= 10000:
            fitness_list.append(find_fitness(pop))
        child.append(child2)
        generations += 1
        # print("Generation: " + str(generations))
        if generations <= 10000:
            fitness_list.append(find_fitness(pop))
        # generations += 2  # increase generation number by two
        if generations % 1000 == 0:
            print("Image will be saved")
            drawer(pop, num_genes)
        # print("Number of generations: " + str(generations))  # debugger
    return_list = chosen_individuals + child  #
    return return_list


def mutation(will_be_mutated, mut_prob, mut_type, gene_number):
    for i in range(len(will_be_mutated)):
        probability = random.uniform(0, 1)  # random probability
        random_gene = random.randint(0, gene_number - 1)
        if probability < mut_prob:  # will be mutated
            if mut_type:  # un-guided
                will_be_mutated[i].genes_list[random_gene].x = random.randint(0, 179)  # random x
                will_be_mutated[i].genes_list[random_gene].y = random.randint(0, 179)  # random y
                will_be_mutated[i].genes_list[random_gene].radii = random.randint(0, 90)  # random radii
                will_be_mutated[i].genes_list[random_gene].red = random.randint(0, 255)  # random red
                will_be_mutated[i].genes_list[random_gene].green = random.randint(0, 255)  # random green
                will_be_mutated[i].genes_list[random_gene].blue = random.randint(0, 255)  # random blue
                will_be_mutated[i].genes_list[random_gene].alpha = random.uniform(0, 1)  # random alpha

            else:  # guided
                # capture parameters
                x = will_be_mutated[i].genes_list[random_gene].x
                y = will_be_mutated[i].genes_list[random_gene].y
                radii = will_be_mutated[i].genes_list[random_gene].radii
                red = will_be_mutated[i].genes_list[random_gene].red
                green = will_be_mutated[i].genes_list[random_gene].green
                blue = will_be_mutated[i].genes_list[random_gene].blue
                alpha = will_be_mutated[i].genes_list[random_gene].alpha
                while True:  # loop until finding correct deviations
                    x_ = random.randint(x - 45, x + 45)
                    y_ = random.randint(y - 45, y + 45)
                    radii_ = random.randint(radii - 10, radii + 10)
                    red_ = random.randint(red - 64, red + 64)
                    green_ = random.randint(green - 64, green + 64)
                    blue_ = random.randint(blue - 64, blue + 64)
                    alpha_ = random.uniform(alpha - 0.25, alpha + 0.25)
                    if (0 <= x_ <= 180) and (0 <= y_ <= 180) and \
                            (0 <= radii_ <= 90) and (0 <= red_ <= 255) and \
                            (0 <= green_ <= 255) and (0 <= blue_ <= 255) and (0 <= alpha_ <= 1):
                        will_be_mutated[i].genes_list[random_gene].x = x_
                        will_be_mutated[i].genes_list[random_gene].y = y_
                        will_be_mutated[i].genes_list[random_gene].radii = radii_
                        will_be_mutated[i].genes_list[random_gene].red = red_
                        will_be_mutated[i].genes_list[random_gene].green = green_
                        will_be_mutated[i].genes_list[random_gene].blue = blue_
                        will_be_mutated[i].genes_list[random_gene].alpha = alpha_
                        break

    return will_be_mutated


def drawer(pop, gene_number):
    best = pop.population_list[0].fitness
    index = 0
    for i in range(1, len(pop.population_list)):
        if pop.population_list[i].fitness > best:
            best = pop.population_list[i].fitness
            index = i
    print("BEST FITNESS: " + str(best))
    sample_ind = pop.population_list[index]
    w_img = white_img()  # white image
    over_img = np.empty_like(w_img)  # initialize overlay numpy array
    sample_ind.sort_radii()  # sort radii of the individual
    for j in range(gene_number):  # iterate over genes
        over_img[:] = w_img  # copy image to overlay
        gene = sample_ind.genes_list[j]  # capture the gene
        over_img = cv2.circle(over_img, (gene.x, gene.y), gene.radii, (gene.red, gene.green, gene.blue),
                              -1)  # draw the circle on overlay
        w_img = (over_img.astype(np.float64) * sample_ind.genes_list[j].alpha + w_img.astype(np.float64) * (
                1 - sample_ind.genes_list[j].alpha)).astype(np.uint8)  # consider uint8 conversion
    save_path = 'img_' + str(generations // 1000) + '.jpg'
    cv2.imwrite(save_path, w_img)
    # cv2.imshow('img', w_img)
    # cv2.waitKey(0)


def find_fitness(pop):
    best = pop.population_list[0].fitness
    for i in range(1, len(pop.population_list)):
        if pop.population_list[i].fitness > best:
            best = pop.population_list[i].fitness
    return best


# THE EVOLUTIONARY ALGORITHM
def main():
    pop = initialize_pop(num_inds, num_genes)
    while generations <= num_generations + 100:
        # print("POPULATION: "+str(len(pop.population_list)))
        for i in range(len(pop.population_list)):
            evaluation(pop.population_list[i])
            # print("EVALUATION: "+str(i))
        adder = mutation(crossover(pop, tm_size, num_genes), mutation_prob, mutation_type, num_genes)
        # print(len(adder))
        for i in range(len(adder)):
            pop.population_list.append(adder[i])
        # print(generations)
        # print("Number of generations: " + str(generations))  # debugger
    plt.plot(x_coordinate[:10000], fitness_list[:10000])
    # naming the x axis
    plt.xlabel('generations')
    # naming the y axis
    plt.ylabel('fitness')
    # giving a title to my graph
    plt.title('Fitness vs. Generations (baseline)')
    # function to show the plot
    plt.savefig('baseline_10000.png')
    plt.clf()  # clear plot
    plt.plot(x_coordinate1000[:1000], fitness_list[:1000])
    # naming the x axis
    plt.xlabel('generations')
    # naming the y axis
    plt.ylabel('fitness')
    # giving a title to my graph
    plt.title('Fitness vs. Generations (baseline)')
    # function to show the plot
    plt.savefig('baseline_1000.png')


if __name__ == "__main__":
    main()
