from Algorithm import Init_run
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from random import randint
import random
import operator
import itertools
import statistics as stats


#############################################################################

#Initialization step


# Initialize sets I and L
# Set number of parts as set I for 1...u
num_parts = 2

# Set number of stores as set L
num_stores = 10

# Failure rates of parts at each store (demand) per hour
# Demand = .02 = 1 part sold every 50 hours
# Low, medium, and high demand represented
#lambda_demand = [.02, .04, .06, .04, .02, .05, .08, .04, .01, .02]
#lambda_demand = [.08, .16, .24, .16, .08, .15, .32, .16, .04, .08]
#lambda_demand = [np.array([1/1, 1/3, 1/2, 1/5, 2/1, 4/3, 2/3, 3/1, 1/10, 1/50])]
lambda_demand = [np.repeat(1/1, 10)]

# Length of the simulation run
t = 100
    
# Create random list of unit cost for each part 
C = [] 
c_i = 30
for j in range(num_parts): 
    c_i = randint(2,30)
    C.append(c_i)
    #C.append(randint(2, 30)) 
    
# Penalty cost
c_p = 100

# Threshold for wait times with tolerance
tol_e = .5
Tol = np.repeat(10, num_stores)
Tlow = np.repeat(Tol-tol_e,num_stores)
Tup = np.repeat(Tol+tol_e,num_stores)    

# Set the upper limits for stocking quantity at each store and DC
# upper limit is 10 for everything
Sil_upper_list = [np.repeat(10,10),np.repeat(10,10)]
Sil_upper = pd.DataFrame(Sil_upper_list, columns = ['1','2','3','4','5','6','7','8','9','10'],
                         index = ['1','2']) 

# upper limit is 20 for the DC
Si0_upper = np.repeat(10, num_parts)
# number of parts stocked at dc
Si0 = [randint(1,Si0_upper[i]) for i in range(num_parts)]
    
# Simulation runs
nought = 25

# weights
w1 = .5
w2 = .5
    

def xln_create(current_solution):
    xln = Init_run.xln_value(current_solution, num_parts, num_stores, lambda_demand, t)
    return xln


#######################################
# step 1 Mating
# Assign objective function value
# Evaluate fitness from problem 2 (p2)
# DC stock kept constant ??????

def p2_objective_func(xln, j, Tol, w1, w2, sol):
    print("p2 objective function")
    dc_cost = 0
    for i in range(num_parts):
        dc_cost = dc_cost + (Si0[i] * C[i])
    
    inv_cost = 0
    sol_j = sol[j]
    inv_cost = sum(sum(C*sol_j.values.T))
    # First portion of the objective function w1
    part_one = w1*(inv_cost+dc_cost)
    
    si_new = xln[j].mean(axis=0)-Tol
    si_new[si_new < 0] = 0
    sum_all_stores =  si_new.sum(axis=0)
    # Penalty for wait time in objective function w2
    part_two = c_p * sum_all_stores * 100
    
    return w1*part_one+w2*part_two
    
#obj = p2_objective_func(xln, 1, Tol, w1, w2)

# Linear ranking of solutions
def linear_ranking_scheme(xln, w1, w2, sol):
    print("Linear ranking scheme")
    M = len(xln)
    fitness_list = {}
    for j in range(M):
        fitness_list[j]=(round(p2_objective_func(xln, j, Tol, w1, w2, sol),2))
        
    fitness_list =  sorted(fitness_list.items(), key=operator.itemgetter(1))
    return fitness_list

# fitness_list = linear_ranking_scheme(xln_start, w1, w2, current_sol)

# Assign probabilities to each solution based on ranking
def prob_linear_rank(rm, w1, w2, xln_start, M, current_sol):
    print("prob linear rank")
    p2_dict = linear_ranking_scheme(xln_start, w1, w2, current_sol)
    n_neg = 2/(rm+1)
    n_pos = (2*rm) / (rm+1)
    p2_prob = {p2_dict[i][0]:(1/M)*(n_neg + (n_pos-n_neg)*(((M-i)-1)/(M-1))) for i in range(len(p2_dict))}

    cum_prob = 0
    p2_cum_prob = {}
    for key, value in p2_prob.items():
        cum_prob = cum_prob + value
        p2_cum_prob[key] = cum_prob
        
    # check that probabilities sum to one: sum(p2_prob.values())
    return p2_dict, p2_prob, p2_cum_prob

# p2_dict, p2_prob, p2_cum_prob = prob_linear_rank(5, w1, w2)     
           
# Selects the parents according to the random number drawn using cumulative probs 
def select_parents(p2_cum_prob, M):
    parents = []
    print("Random num : Parent probability")
    for j in range(M):
        random_num = random.uniform(0,1)
        print(random_num)
        first_index= list(p2_cum_prob.keys())[0]
        if random_num < p2_cum_prob[first_index]:
            parent = p2_cum_prob[first_index]
        else:
            parent = min([p2_cum_prob[k] for k in range(len(p2_cum_prob)) if p2_cum_prob[k] >= random_num])
        print(random_num," : ", parent)
        for key, value in p2_cum_prob.items(): 
            if parent == value: 
                parent_ind = key

        
        parents.append(parent_ind)
    return parents

# parent = select_parents(p2_cum_prob, M)

# Pairs all parents and remove duplicate entries
def select_parent_pairs(parent):
    print("select parents")
    pairs_raw = list(itertools.combinations(parent, 2))
    # Remove duplicates
    pairs = []
    [pairs.append(x) for x in pairs_raw if x not in pairs]
    count_pair = len(pairs)
    return pairs, count_pair

# pair, num_pair = select_parent_pairs(parent)

###########################################

# Mates the parent pairs to produce offspring solutions
def mating(parent_1, parent_2, num_stores, num_parts, method='Single'):
    print("mating")
    gene_len = num_stores*num_parts
    offspring = []
    if method == 'Single':
        pivot_point = randint(1, gene_len)
        offspring_1 = np.concatenate([parent_1[0:pivot_point],parent_2[pivot_point:]])
        offspring_2 = np.concatenate([parent_2[0:pivot_point],parent_1[pivot_point:]])
        offspring= [offspring_1, offspring_2]
    else: # Double pivot
        pivot_one = randint(1,gene_len-1)
        pivot_two = randint(1,gene_len)
        print("Pivots: ", pivot_one, " and ", pivot_two)
        offspring_1 = np.concatenate([parent_1[0:pivot_one],parent_2[pivot_one:]]).reshape(num_parts, num_stores)
        print(offspring_1)
        offspring_2 = np.concatenate([parent_2[0:pivot_one],parent_1[pivot_one:]]).reshape(num_parts, num_stores)
        print(offspring_2)
        offspring_3 = np.concatenate([parent_1[0:pivot_two],parent_2[pivot_two:]]).reshape(num_parts, num_stores)
        print(offspring_3)
        offspring_4 = np.concatenate([parent_2[0:pivot_two],parent_1[pivot_two:]]).reshape(num_parts, num_stores)
        print(offspring_4)
        offspring = [pd.DataFrame(offspring_1), pd.DataFrame(offspring_2), pd.DataFrame(offspring_3), pd.DataFrame(offspring_4)]
    return offspring

def p1_objective_func(j, sol):
    print("p1 objective function")
    dc_cost = 0
    for i in range(num_parts):
        dc_cost = dc_cost + (Si0[i] * C[i])
    
    inv_cost = 0
    sol_j = sol[j]
    inv_cost = sum(sum(C*sol_j.values.T))
    # First portion of the objective function w1
    part_one = (inv_cost+dc_cost)
    return part_one

def linear_ranking(sol):
    print("Linear ranking scheme")
    M = len(sol)
    fitness_list = {}
    for j in range(M):
        fitness_list[j]=(round(p1_objective_func(j, sol),2))
        
    fitness_list =  sorted(fitness_list.items(), key=operator.itemgetter(1))
    return fitness_list

# For each parent pair, the offspring are created and ranked
# Top 2 children are returned to be added to the population
def crossover(pair, i, solution_space):
    print("crossover")
    top_children = []
    parent_1_ind = pair[i][0]
    parent_2_ind = pair[i][1]
    parent_1 = np.array(solution_space[parent_1_ind]).flatten()
    parent_2 = np.array(solution_space[parent_2_ind]).flatten()
    offspring = mating(parent_1, parent_2, num_stores, num_parts, 'Double')
    
    '''
    S_dict = {}
    for s in range(len(offspring)):
        xln_df = pd.DataFrame(xln_create(offspring[s]) for i in range(nought))
        print(s, " iteration in offspring run")
        S_dict[s] = xln_df
    print(S_dict)
    rank = linear_ranking_scheme(S_dict, w1, w2, offspring)
    print("rank ", rank)
    '''
    rank = linear_ranking(offspring)
        
    top_child_one = rank[0][0]
    top_child_two = rank[1][0]
        
    top_children.append(offspring[top_child_one])
    top_children.append(offspring[top_child_two])
    return top_children

# cross = crossover(pair, 1, current_sol)

# Top 2 children are added to the population for each parent pair
def crossover_population(pair, solution_space):
    crossover_pop ={}
    j = 0
    for i in range(len(pair)):
        print("Pair")
        print(pair[i])
        top_2 = crossover(pair, i, solution_space)
        crossover_pop[j] = top_2[0]
        crossover_pop[j+1] = top_2[1]
        j = j+2
    return crossover_pop
 
#new_pop = crossover_population(pair, current_sol)

def mutation(offspring):
    print("mutation")
    mutation_rate = random.uniform(.05, .2)
    offspring = np.ravel(offspring)
    for i in range(len(offspring)):
        random_num = random.uniform(0,1)
        if random_num < mutation_rate:
            print("mutate")
            offspring[i] = randint(1, np.array(Sil_upper).flatten()[i])
    return pd.DataFrame(offspring.reshape(2,10))

def build_new_pop(pair, current_sol):
    new_pop = {}
    elite_pop = {}
    final_pop = {}
    cross_pop = crossover_population(pair, current_sol)
    for j in range(len(cross_pop)):
        offspring = mutation(cross_pop[j])
        new_pop[j] = offspring
    
    S_dict = {}
    for s in range(len(new_pop)):
        xln_df = pd.DataFrame(xln_create(new_pop[s]) for i in range(nought))
        print(s, " iteration in offspring run")
        S_dict[s] = xln_df
    rank = linear_ranking_scheme(S_dict, w1, w2, new_pop)
    print("rank ", rank)
    
    top_rank = rank[0][0]
    elite_pop[1] = new_pop[top_rank]
    top = min(20, len(new_pop))
    for k in range(top):
        final_pop[k] = new_pop[rank[k][0]]
        
    return elite_pop, final_pop
    
# el = build_new_pop()
    
def main():
    print("Main")
    # Initialize the algorithm with parameters, constraints, and thresholds

    total = num_parts*num_stores

    # Response time from DC to each store in hours as set 
    # from uniform(1,5) days distribution
    transport_times = []
    for i in range(num_stores):
        transport_times.append(random.uniform(1,5))
        
    alpha = .95
    k = 1 #num of solutions that are being checked
    v = num_stores

    # Randomly pick M solutions from the soution space
    
    total_sol = 100
    M = 10
    
    # Example of one stocked store
    one_store = Init_run.stock_store(num_stores, num_parts, Sil_upper)
    
    
    
    # Create solution space (100 solutions)
    all_solutions = Init_run.solution_space_total(total_sol, num_stores, num_parts, Sil_upper)
    
    # Pick M solutions from the space
    m_solutions = Init_run.pick_M_solutions(all_solutions, M, total_sol)
    
    # For each solution, take nought observations and find Xln
    # Xln is for each store, simulation run
    
    # Make vector of Qile as number of sales for part i at store l 
    # in time t under one simulation run
    
    from numba import jit, prange
    from joblib import Parallel, delayed
    
    #xln_first = Init_run.xln_sim_runs(M, num_parts, num_stores, lambda_demand, t, nought, m_solutions)
    
    def sim_runs_parallel(x):
        print("sim_runs_parallel")
        xln_result = Init_run.xln_sim_runs(M, num_parts, num_stores, lambda_demand, t, nought, m_solutions)
        return xln_result

    results = Parallel(n_jobs=-1, backend="threading")(map(delayed(sim_runs_parallel),[0,0]))
    #print(results)
    ######################################################################
    print("xln_start")
    print(results[0])
    xln_start = results[0]
    
    print(xln_start)
    
    
    current_sol = m_solutions
    M = len(current_sol)
    parents = []
    offspring = []
    elite_pop = {}
    
    fit_list = []
    fitness_score = []
    
    print("===========================================")
    print(xln_start[0][0])
    for j in range(M):
        fitness = p2_objective_func(xln_start, j, Tol, w1, w2, current_sol)
        fit_list.append(fitness)
    fitness_avg = stats.mean(fit_list)
    print("Starting fitness: ", fitness_avg)
    fitness_score.append(fitness_avg)
    
    fitness_list = linear_ranking_scheme(xln_start, w1, w2, current_sol)
    
    p2_dict, p2_prob, p2_cum_prob = prob_linear_rank(5, w1, w2, xln_start, M, current_sol)   

    parent = select_parents(p2_cum_prob, M)
    
    pair, num_pair = select_parent_pairs(parent)
    
    elite_pop, new_pop = build_new_pop(pair, current_sol)
    
    current_sol = new_pop
    xln_set = Init_run.xln_sim_runs(M, num_parts, num_stores, lambda_demand, t, nought, current_sol)
    fit_list = []
    M = len(current_sol)
    for j in range(M):
        fitness = p2_objective_func(xln_set, j, Tol, w1, w2, current_sol)
        fit_list.append(fitness)
    fitness_avg = stats.mean(fit_list)
    print("Starting fitness: ", fitness_avg)
    fitness_score.append(fitness_avg)
    
    for i in range(50):
        if (i==0):
            M = min(len(current_sol), 20)
            current_sol = {key:value for key,value in list(new_pop.items())[0:20]}
            xln_set = {key:value for key,value in list(xln_set.items())[0:20]}
        #xln_set = {key:value for key,value in list(xln_set.items())[0:50]}
        #xln_set = Init_run.xln_sim_runs(M, num_parts, num_stores, lambda_demand, t, nought, current_sol)
        
        # fitness_list = linear_ranking_scheme(xln_set, w1, w2, current_sol)
    
        p2_dict, p2_prob, p2_cum_prob = prob_linear_rank(5, w1, w2, xln_set, M, current_sol)   

        parent = select_parents(p2_cum_prob, M)
    
        pair, num_pair = select_parent_pairs(parent)
    
        elite_pop, new_pop = build_new_pop(pair, current_sol)
        
        current_sol = {key:value for key,value in list(new_pop.items())[0:20]}
        M = min(len(current_sol), 20)
        xln_set = Init_run.xln_sim_runs(M, num_parts, num_stores, lambda_demand, t, nought, current_sol)
        fit_list = []
        for j in range(M):
            fitness = p2_objective_func(xln_set, j, Tol, w1, w2, current_sol)
            fit_list.append(fitness)
        print("Starting fitness: ", stats.mean(fit_list))
        fitness_score.append(stats.mean(fit_list))
    
    return fitness_score, elite_pop, results[0], new_pop

    
if __name__ == "__main__":
    fit = main()
