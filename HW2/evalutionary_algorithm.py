# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:08:16 2024

@author: defne
"""

"""
Initialize population with <num_inds> individuals each having <num_genes> genes
While not all generations (<num_generations>) are computed:
    Evaluate all individuals
    Select individuals
    Do crossover on some individuals
    Mutate some individuals
"""

import numpy as np
import random 
import cv2
import copy
import matplotlib.pyplot as plt

#Invidiual has one chromosome. Each gene in a chromosome represents one circle to be drawn.
#Each gene has at least 7 values: (x,y), r (radius), colors (green, blue,alpha) and A

#Define global variables to be used
source_path = 'C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW2\\painting.png'

source_image = cv2.imread(source_path)
image_shape = np.shape(source_image)

IMG_WIDTH = image_shape[0]
IMG_HEIGHT = image_shape[1]

upper_limit = min(IMG_HEIGHT, IMG_WIDTH) // 4

class Gene:
    def __init__(self):
        while True: 
            self.x = random.randint(-2*IMG_WIDTH, 2*IMG_WIDTH) #center x-coordinate
            self.y = random.randint(-2*IMG_HEIGHT, 2*IMG_HEIGHT) #center y-coordinate
            self.radius = random.randint(1, min(IMG_HEIGHT, IMG_WIDTH)//4) #radius
            self.R = random.randint(0, 255) # R
            self.G = random.randint(0, 255) # G
            self.B = random.randint(0, 255) # B
            self.alpha = random.uniform(0, 1) # Alpha
            
            if not self.valid_circle():
                #self.print_gene()
                break #Exit the loop
    
    def valid_circle(self):
        if (self.x + self.radius <= 0) or (self.x - self.radius >= IMG_WIDTH) or (self.y + self.radius <= 0) or (self.y - self.radius >= IMG_HEIGHT):
            #Outside the image
            #Should be reinitialized
            return True
        else:
            return False
    
    def correct_gene(self):
        
        # Correct gene attributes to valid values if necessary
        self.x = max(-2 * IMG_WIDTH, min(2 * IMG_WIDTH, self.x))
        self.y = max(-2 * IMG_HEIGHT, min(2 * IMG_HEIGHT, self.y))
        self.radius = max(1, min(min(IMG_HEIGHT, IMG_WIDTH) // 4, self.radius))
        self.R = max(0, min(255, self.R))
        self.G = max(0, min(255, self.G))
        self.B = max(0, min(255, self.B))
        self.alpha = max(0, min(1, self.alpha))
        
        print('corrected somehow')
    
    def print_gene(self):
        print(f"  x: {self.x}")
        print(f"  y: {self.y}")
        print(f"  radius: {self.radius}")
        print(f"  R: {self.R}")
        print(f"  G: {self.G}")
        print(f"  B: {self.B}")
        print(f"  alpha: {self.alpha}")
        print(f"  Color: ({self.B}, {self.G}, {self.R})")


class Individual:
    def __init__(self, num_genes, chromosome = []):
        self.num_genes = num_genes
        self.chromosome = chromosome if chromosome else [Gene() for i in range(num_genes)] #Individual has one chromosome, each gene in a chromosome represents one circle to be drawn.
        self.fitness = 0 #fitness value
        
        # if len(self.chromosome) == 0:
        #     for _ in range(num_genes):
        #         self.chromosome.append(Gene()) #a gene object is created and added to the chromosome
        
    def sort_genes(self):
        self.chromosome = sorted(self.chromosome, key=lambda x: x.radius , reverse = True)    
    # def initialize_random(self):
    #     #Initialize genes randomly within image boundaries
    #     for _ in range(self.num_genes):
            
    #         self.chromosome.append(Gene()) #a gene object is created and added to the chromosome

    # Draw the individual circle  
    def draw(self):
        #Draw circles on the image
        image_new = 255 * np.ones(image_shape, np.uint8) #Initializing overlay as white image
        
        self.sort_genes() #genes are sorted according to their radius
        
        for gene in self.chromosome:
            #A deepcopy of the whiteFrame is created
            overlay = copy.deepcopy(image_new)
            
            color = (gene.B, gene.G, gene.R)  # OpenCV uses BRG format
            # a circle is created on to the deepcopy of the overlay
            #print(f"Color: {color}")
            cv2.circle(overlay, (gene.x,gene.y), gene.radius, color, -1) #A filled circle
            # there is a problem here fix please
            cv2.addWeighted(overlay, gene.alpha, image_new, 1 - gene.alpha, 0, image_new)
            
        return image_new
    
    # Evaluate the fitness of the individual
    def evaluate_fitness(self):
        # Draw circles and calculate fitness
        # Implementation of drawing circles and calculating fitness goes here
        image = self.draw()
        image = np.array(image, np.int64)
        source_np = np.array(source_image, np.int64)
        
        squared_diff = np.square(source_np - image)
        self.fitness = -np.sum(squared_diff)
    
    def print_genes(self):
        for idx, gene in enumerate(self.chromosome):
            print(f"Gene {idx + 1}:")
            print(f"  x: {gene.x}")
            print(f"  y: {gene.y}")
            print(f"  radius: {gene.radius}")
            print(f"  R: {gene.R}")
            print(f"  G: {gene.G}")
            print(f"  B: {gene.B}")
            print(f"  alpha: {gene.alpha}")
    # Mutate the individuals
    def mutate_gene(self, mutation_prob, mutation_type, radius_deviation, RGB_deviation, alpha_deviation):
        # if random.random() >= mutation_prob:
        #     return
        
        #Randomly change random genes
        while random.random() < mutation_prob:
            
            #random_gene = random.choice(self.chromosome) #A random gene is selected
            # Randomly select a gene to be mutated
            index_to_mutate = random.randint(0, self.num_genes - 1)
            random_gene = self.chromosome[index_to_mutate]
            
            if mutation_type == 'Unguided':
                #print('there is mutation unguided')
                
                while True:
                    random_gene.x = random.randint(-2*IMG_WIDTH, 2*IMG_WIDTH) #center x-coordinate
                    random_gene.y = random.randint(-2*IMG_HEIGHT, 2*IMG_HEIGHT) #center y-coordinate
                    random_gene.radius = random.randint(1, min(IMG_HEIGHT, IMG_WIDTH)//4) #radius
                    random_gene.R = random.randint(0, 255) # R
                    random_gene.G = random.randint(0, 255) # G
                    random_gene.B = random.randint(0, 255) # B
                    random_gene.alpha = random.uniform(0, 1) # Alpha
                    
                    if not random_gene.valid_circle():
                        #Replace the mutated gene to its place
                        #self.chromosome[index_to_mutate] = copy.deepcopy(random_gene)
                        break             
                
            elif mutation_type == 'Guided':
                
                x_backup = random_gene.x
                y_backup = random_gene.y
                radius_backup = random_gene.radius
                
                R_backup = random_gene.R
                G_backup = random_gene.G
                B_backup = random_gene.B
                alpha_backup = random_gene.alpha
                
                lower_x_limit = x_backup - IMG_WIDTH//4 + 1
                upper_x_limit = x_backup + IMG_WIDTH//4 - 1
                lower_y_limit = y_backup - IMG_HEIGHT//4 + 1
                upper_y_limit = y_backup + IMG_HEIGHT//4 - 1
                
                
                while True:
                    #Deviate the x,y,radius,R,G,B,A
                                                  
                    random_gene.x = random.randint(lower_x_limit, upper_x_limit)
                    random_gene.y = random.randint(lower_y_limit, upper_y_limit)
                    
                    
                    
                        
                    if (radius_backup - radius_deviation < 1 and radius_backup + radius_deviation <= upper_limit): 
                        random_gene.radius = random.randint(1, radius_backup + radius_deviation)
                    elif (radius_backup - radius_deviation >= 1 and radius_backup + radius_deviation > upper_limit):
                        random_gene.radius = random.randint(radius_backup - radius_deviation, upper_limit)
                    elif (radius_backup - radius_deviation < 1 and radius_backup + radius_deviation > upper_limit):
                        random_gene.radius = random.randint(1, upper_limit)
                    else:
                        random_gene.radius = random.randint(radius_backup - radius_deviation, radius_backup + radius_deviation)
                   
                    
                    if not random_gene.valid_circle():
                        #Replace the mutated gene to its place
                        #random_gene.print_gene()
                        
                        upper_limit_rgb = 255
                        if (R_backup - RGB_deviation < 0 and R_backup + RGB_deviation <= upper_limit_rgb): 
                            random_gene.R = random.randint(0, R_backup + RGB_deviation)
                        elif (R_backup - RGB_deviation >= 0 and R_backup + RGB_deviation > upper_limit_rgb):
                            random_gene.R = random.randint(R_backup - RGB_deviation, upper_limit_rgb)
                        else:
                            random_gene.R = random.randint(R_backup - RGB_deviation, R_backup + RGB_deviation)
                            
                        if (G_backup - RGB_deviation < 0 and G_backup + RGB_deviation <= upper_limit_rgb): 
                            random_gene.G = random.randint(0, G_backup + 64)
                        elif (G_backup - RGB_deviation >= 0 and G_backup + RGB_deviation > upper_limit_rgb):
                            random_gene.G = random.randint(G_backup - RGB_deviation, upper_limit_rgb)
                        else:
                            random_gene.G = random.randint(G_backup - RGB_deviation, G_backup + RGB_deviation)
                            
                        if (B_backup - RGB_deviation < 0 and B_backup + RGB_deviation <= upper_limit_rgb): 
                            random_gene.B = random.randint(0, B_backup + RGB_deviation)
                        elif (B_backup - RGB_deviation >= 0 and B_backup + RGB_deviation > upper_limit_rgb):
                            random_gene.B = random.randint(B_backup - RGB_deviation, upper_limit_rgb)
                        else:
                            random_gene.B = random.randint(B_backup - RGB_deviation, B_backup + RGB_deviation)
                            
                        
                        if (alpha_backup - alpha_deviation < 0 and alpha_backup + alpha_deviation <= 1): 
                            random_gene.alpha = random.uniform(0, alpha_backup + alpha_deviation)
                        elif (alpha_backup - alpha_deviation >= 0 and alpha_backup + alpha_deviation > 1):
                            random_gene.alpha = random.uniform(alpha_backup - alpha_deviation, 1)
                        else:
                            random_gene.alpha = random.uniform(alpha_backup - alpha_deviation, alpha_backup + alpha_deviation)
                        
                        # upper_limit_rgb = 255
                        # if (R_backup - 64 < 0 and R_backup + 64 <= upper_limit_rgb): 
                        #     random_gene.R = random.randint(0, R_backup + 64)
                        # elif (R_backup - 64 >= 0 and R_backup + 64 > upper_limit_rgb):
                        #     random_gene.R = random.randint(R_backup - 64, upper_limit_rgb)
                        # else:
                        #     random_gene.R = random.randint(R_backup - 64, R_backup + 64)
                            
                        # if (G_backup - 64 < 0 and G_backup + 64 <= upper_limit_rgb): 
                        #     random_gene.G = random.randint(0, G_backup + 64)
                        # elif (G_backup - 64 >= 0 and G_backup + 64 > upper_limit_rgb):
                        #     random_gene.G = random.randint(G_backup - 64, upper_limit_rgb)
                        # else:
                        #     random_gene.G = random.randint(G_backup - 64, G_backup + 64)
                            
                        # if (B_backup - 64 < 0 and B_backup + 64 <= upper_limit_rgb): 
                        #     random_gene.B = random.randint(0, B_backup + 64)
                        # elif (B_backup - 64 >= 0 and B_backup + 64 > upper_limit_rgb):
                        #     random_gene.B = random.randint(B_backup - 64, upper_limit_rgb)
                        # else:
                        #     random_gene.B = random.randint(B_backup - 64, B_backup + 64)
                            
                        
                        # if (alpha_backup - 0.25 < 0 and alpha_backup + 0.25 <= 1): 
                        #     random_gene.alpha = random.uniform(0, alpha_backup + 0.25)
                        # elif (alpha_backup - 0.25 >= 0 and alpha_backup + 0.25 > 1):
                        #     random_gene.alpha = random.uniform(alpha_backup - 0.25, 1)
                        # else:
                        #     random_gene.alpha = random.uniform(alpha_backup - 0.25, alpha_backup + 0.25)
                            
                            
                        #self.chromosome[index_to_mutate] = copy.deepcopy(random_gene)
                        break
                    
class Population:
    
    def __init__(self, num_inds, num_genes):
        self.num_inds = num_inds
        self.num_genes = num_genes
        self.population = [Individual(num_genes) for _ in range(num_inds)] #start the population   
        
        #self.initialize_population()
        # for _ in range(self.num_inds):
        #     self.population.append(Individual(self.num_genes))
        
    # def initialize_population(self):
    #     for _ in range(self.num_inds):
    #         #individual = Individual(self.num_genes)
    #         #individual.initialize_random()
    #         self.population.append(Individual(self.num_genes))
    
    def selection(self, frac_elites, frac_parents, tm_size):
        
        num_elites = int(self.num_inds * frac_elites) # num elites are chosen
        num_parents = int(self.num_inds * frac_parents) # num of parents are chosen
        
        #number of parents should be even 
        if num_parents % 2 == 1:
            num_parents += 1 # even
        
        #Sort the indviduals based on fitness
        self.population.sort(key=lambda x: x.fitness, reverse = True)
        
        #Select elites
        elites = self.population[:num_elites]
        
        other_ind = self.population[num_elites:]
        
        non_elites = []
        
        # the selection of other individuals are done with tournament selection
        for _ in range(len(other_ind)):
            tournament = random.sample(other_ind , min(tm_size, len(other_ind)))
            winner = max(tournament, key=lambda x: x.fitness)
            non_elites.append(winner)
            
        #another tournament selection for the parents now
        # The selection of parent individuals 
        # Same individual can still win the tournament as a parent.
        parents = []
        
        non_elites.sort(key = lambda x: x.fitness, reverse = True)
        parents = non_elites[:num_parents] #best of the non_elites are taken as the parents
        
        # Remove the selected parents from non_elites using list comprehension
        non_elites = non_elites[num_parents:]
        
        return elites, non_elites, parents
        
    def crossover(self, parents):
        
        childeren = []
        
        # Pair up parents
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i+1]
            
            child1_chromosome = []
            child2_chromosome = []
            
            for j in range(self.num_genes):
                # Exchange of gene is calculated individually with equal probability
                if random.random() < 0.5:
                    child1_chromosome += [copy.deepcopy(parent1.chromosome[j])] #child1 gets the gene from parent1
                    child2_chromosome += [copy.deepcopy(parent2.chromosome[j])] #child2 gets the gene from parent2
                    
                else:
                    child1_chromosome += [copy.deepcopy(parent2.chromosome[j])] #child1 gets the gene from parent2
                    child2_chromosome += [copy.deepcopy(parent1.chromosome[j])] #child2 gets the gene from parent1
                
            #After the exchange the childs become individuals with the settled genes
            child1 = Individual(self.num_genes, child1_chromosome) #Individual object of child1
            child2 = Individual(self.num_genes, child2_chromosome) #Individual object of child2
            child1.evaluate_fitness()
            child2.evaluate_fitness()
            
            childeren.append(child1)
            childeren.append(child2)
        
        #Parents are lost
        return childeren
    def mutate_pop(self, population, mutation_prob, mutation_type, generation):
        #Mutate some individuals
        for individual in population:
            if generation <= 3000:
                individual.mutate_gene(mutation_prob, mutation_type, 15, 100, 0.4) #deviation of the radius is raduced
            elif generation > 3000 and generation <= 7000:
                individual.mutate_gene(mutation_prob, mutation_type, 10, 64, 0.25) #deviation of the radius is reduced
            elif generation > 7000:
                individual.mutate_gene(mutation_prob, mutation_type, 5, 32, 0.1) #deviation of the radius is reduced
                
    def evolutionary_algorithm(self, num_generations, frac_elites, frac_parents, tm_size, mutation_prob, mutation_type):
        
        fitness_history = []
        image_filenames = [] # Add this line to keep track of the saved image filenames
        mutation_prob = 0.6

        for generation in range(num_generations):
            #print(f"Generation {generation + 1}")
            
            if(generation <= 1000):
                if (generation + 1) % 100 == 0:
                    #high probability for higher diversity
                    mutation_prob = mutation_prob - (0.2/10)
            elif (generation > 1000) and (generation < 5000):
                if (generation + 1) % 1000 == 0:
                    #reduces the mutation probability with small and reaches to 0.1 after generation 5000
                    mutation_prob = mutation_prob - (0.1/4)
                    
            #Evaluate all individuals
            for individual in self.population:
                individual.evaluate_fitness()
                
            #select individuals
            elites, non_elites, parents = self.selection(frac_elites, frac_parents, tm_size)
            
            #Do crossover on some individuals
            childeren = self.crossover(parents)
            
            non_elites += childeren
            
            
            #Mutate some individuals
            self.mutate_pop(non_elites, mutation_prob, mutation_type, generation)
            
            for i in non_elites:
                i.evaluate_fitness()
            
            #reassigning the population
            self.population = elites + non_elites
            
            best_individual = max(self.population, key=lambda x : x.fitness)
            
            # Store fitness of the best individual
            fitness_history.append(best_individual.fitness)
            
            if (generation + 1) % 1000 == 0:
                filename = f'best_individual_generation_{generation + 1}.png'
                print(f"Fitness of best individual at generation {generation + 1}: {best_individual.fitness}")
                best_individual_image = best_individual.draw()
                cv2.imwrite(filename, best_individual_image)
                image_filenames.append(filename) # Save the filename
        
        # # Plot fitness history
        # plt.plot(range(1, num_generations + 1), fitness_history)
        # plt.xlabel('Generation')
        # plt.ylabel('Fitness')
        # plt.title('Fitness Plot from Generation 1 to Generation 10000')
        # plt.show()
        
        # plt.plot(range(1000, num_generations + 1), fitness_history[999:])
        # plt.xlabel('Generation')
        # plt.ylabel('Fitness')
        # plt.title('Fitness Plot from Generation 1000 to Generation 10000')
        # plt.show()
        
        best = max(self.population, key=lambda x: x.fitness)
        return best, image_filenames, fitness_history # Return the collected image filenames along with the best individual
    
num_inds = 20
num_genes = 50
tm_size = 5
frac_elites = 0.2
frac_parents = 0.6
mutation_prob = 0.2
mutation_type = 'Guided'
                

pop = Population(num_inds, num_genes)
#pop.initialize_population()
best, image_filenames, fitness_history = pop.evolutionary_algorithm(10000, frac_elites, frac_parents, tm_size, mutation_prob, mutation_type)

def plot_evolution(images, parameters):
    num_rows = 2
    num_cols = 5
    #plt.figure(fig_size=(24,12))
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20,5)) # Adjust the figsize based on your requirement
    
    for i, img_path in enumerate(images):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        row = i // num_cols  # Calculate row index
        col = i % num_cols   # Calculate column index
        axs[row, col].imshow(img)
        axs[row, col].set_title(f'Generation: {(i + 1) * 1000}') # Title with generation
        axs[row, col].axis('off') # Hide axis
        
    plt.suptitle(f"Evolution with Parameters: {parameters}") # Main title with parameter
    plt.tight_layout()
    plt.show()

def plot_fitness(fitness_history, num_generations, parameter):
    #plt.figure(fig_size=(24,12))
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Adjust the figsize to make the figure larger
    
    # Plot for the entire range of generations
    axs[0].plot(range(1, num_generations + 1), fitness_history)
    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Fitness')
    axs[0].set_title('Fitness over All Generations')
    axs[0].grid(linestyle = "dashed")
    
    # Plot for the range from 1000th generation to the last
    axs[1].plot(range(1000, num_generations + 1), fitness_history[999:])
    axs[1].set_xlabel('Generation')
    axs[1].set_ylabel('Fitness')
    axs[1].set_title('Fitness from Generation 1000 to Last Generation')
    axs[1].grid(linestyle = "dashed")

    
    # Add a main title for the whole figure
    plt.suptitle(f'Fitness Evaluation Across Generations for {parameter}')
    
    # Show plot with appropriate layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect if the title overlaps with the subplots
    plt.show()
    
# Call the plotting function with the collected filenames and parameters
plot_evolution(image_filenames, 'Adaptive Mutation Rate and Adaptive Deviation')
plot_fitness(fitness_history, 10000, 'Adaptive Mutation Rate and Adaptive Deviation')

best_image = best.draw()
best.evaluate_fitness() 
filename = 'best_individual_generation_10000_adaptive_mutation_rate_and_deviation.png'
image_path = 'C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE449\\HW2\\' + filename
cv2.imwrite(filename, best_image)


img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
plt.figure(figsize=(10, 5))  # Set the figure size as needed
plt.imshow(img)
plt.title(f'Generation: 10000 - Adaptive_mutation rate and Deviation, Fitness: {best.fitness}')
plt.axis('off')  # Hide axis

plt.show()
print("Fitness:", best.fitness)

# cv2.namedWindow('image_new', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image_new', 200, 200)
# cv2.imshow('image_new', best_image)

# cv2.waitKey(0) #Waiting for a key press to close the window
# cv2.destroyAllWindows() #Close the window        