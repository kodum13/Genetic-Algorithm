import random
from random import randint
import pandas as pd
import numpy as np

# Follows algorithm from the paper:
# A simulation-based decision support system for a multi-echelon inventory problem with service level constraints
# By Shing ChihTsai    Chung HungLiu

print("Init_run")
###############################################################################

# Sets stocking level for each part at all stores
# Takes random integer between 0 and the upper limit inclusive
def stock_store(num_stores, num_parts, Sil_upper):
    stock_level = pd.DataFrame(np.nan, index=[1,2], 
                           columns=['1','2','3','4','5','6','7','8','9','10'])
    for i in range(num_parts):
        for j in range(num_stores):
            stock_level.iloc[i,j] = randint(0,Sil_upper.iloc[i,j])
    return stock_level

# Expected demand over total simulation run time
# Takes the mean lambda (parts sold per day) * time t
def exp_demand(lam_dem, t, num_stores, num_parts):
    exp_dem = pd.DataFrame(np.nan, index=[1,2], 
                           columns=['1','2','3','4','5','6','7','8','9','10'])
    for i in range(num_parts):
        for j in range(num_stores):
            exp_dem.iloc[i,j] = lam_dem[0][j]*t
    return exp_dem

################################################################################

# Get sales time and count of sales of one part at one store
def sales_simulation(exp_mean, t):
    time, num_sales = 0, 0
    sales_time = []
    time_of_sales = np.random.exponential(exp_mean)
    time = time + time_of_sales
    while time < t:
        sales_time.append(time)
        num_sales = num_sales + 1
        time_of_sales = np.random.exponential(exp_mean)
        time = time + time_of_sales
    return sales_time, num_sales

# Simulate the replenishment based on the sales
# Sales initiates a replacement from DC (could change to scheduled replen)
# Builds discrete simulation of sales and replenishment, tracks stock level
def sim_replen(S, store, part, sales_list, Qile):
    # Base stocking level for store and part
    base = S.iloc[part, store]
    replen = []
    for i in range(Qile):
        # Get time the part is replenished at for store and part
        rep_time = random.uniform(1,5) + sales_list[i]
        replen.append(rep_time)
    # All transaction: sale and replenishment times
    sim_list = sorted(sales_list + replen, reverse=False)
    base_list = [int(base)] * len(sim_list)
    stock_list = np.column_stack((sim_list, base_list))
    # Add starting level as time 0 with base stocking level
    starting_list = np.array([[0,int(base)]])
    stock_level = np.concatenate((starting_list, stock_list), axis=0)
    
    for j in range(1,len(sim_list)):
        if stock_level[j][0] not in replen: # represents a sale
            stock_level[j][1] = stock_level[j-1][1] - 1
        else: # replenishes stock
            stock_level[j][1] = stock_level[j-1][1] + 1
    return stock_level, base, replen

# Calculates the wait time given the discrete simulation of sales/replen
def find_wait_time(stock_level, replen, sales_list,base, mean):
    wait_time = []
    for i in range(1,len(stock_level)):
        if stock_level[i][0] in sales_list and stock_level[i][1] > 0:
            repair_time = np.random.exponential(mean)
            wait_time.append(repair_time)
        elif stock_level[i][0] in sales_list and stock_level[i][1] <= 0:
            get_next_replen = min([k for k in replen if k > stock_level[i][0]])
            repair_time = (get_next_replen-stock_level[i][0]) + np.random.exponential(mean)
            wait_time.append(repair_time)
        else:
            pass
    return wait_time

################################################################################
    
# Create all the solution space
def solution_space_total(total_sol, num_stores, num_parts, Sil_upper):
    solution_space = {}

    for i in range(total_sol):
        #get singular solution from stock_store
        solution_space[i] = stock_store(num_stores, num_parts, Sil_upper)

    return solution_space
        
# Pick M solutions from solution space
def pick_M_solutions(all_solutions, M, total_sol):
    sol_list = (list(range(0,total_sol-1)))
    random.shuffle(sol_list)
    m_sol = {}
    for k in range(M):
        m_sol[k] = all_solutions[sol_list[k]]
  
    return m_sol

# For each solution, take nought observations and find Xln
# Xln is for each store, simulation run

# Make vector of Qile as number of sales for part i at store l 
# in time t under one simulation run

def get_Qile_Wile(S, l, num_parts, lambda_demand, t):
    sum_per_store = 0
    wait_total = 0
    for j in range(num_parts):
        # gets sales over time t from store l for part i
        sales_list, Qile = sales_simulation(lambda_demand[0][l], t/10)
        # denominator Qile for all parts at store l
        sum_per_store = sum_per_store + Qile
        # gets replen schedule from store l for part i
        stock_level, base, replen = sim_replen(S, l, j, sales_list, Qile)
        # gets wait time for store l and part i
        wait_list = find_wait_time(stock_level, replen, sales_list, base, mean=10)
        wait_total = sum(wait_list) + wait_total
    # returns the numerator W, and denominator Q for Xln
    return wait_total, sum_per_store

def xln_value(S, num_parts, num_stores, lambda_demand, t):
    q_store = []
    w_store = []
    xln = []
    for l in range(num_stores):
        W, Q = get_Qile_Wile(S, l, num_parts, lambda_demand, t)
        q_store.append(Q)
        w_store.append(W)
        if Q == 0:
            xln.append(0)
        else:
            xln.append(W/Q)
        
    return xln
    
def xln_sim_runs(M, num_parts, num_stores, lambda_demand, t, nought, m_solutions):
    print("xln_sim_runs")
    M_local = len(m_solutions)
    S_dict = {}
    for s in range(M_local):
        xln_df = pd.DataFrame(xln_value(m_solutions[s], num_parts, num_stores, lambda_demand, t) for i in range(nought))
        print(s, " iteration in xln_sim_runs")
        S_dict[s] = xln_df
    return S_dict
