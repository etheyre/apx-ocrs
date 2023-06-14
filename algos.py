import numpy as np, copy, math, multiprocessing as mp, os, time, datetime
# gurobipy as gp, 
import statistics as stats, random, itertools as itt
#from gurobipy import GRB
import scipy.optimize as sopt

eps = 0.1

def ilog(x):
	return np.floor(np.log2(x)/np.log2(1+eps)).astype(int)

def argmin(l, f):
	# warning: explodes if the list l is empty
	curr_min = l[0]
	curr_idx = 0
	
	for i, x in enumerate(l):
		v = f(i, x)
		if v < curr_min:
			curr_min = v
			curr_idx = i
			
	return curr_idx

def lightest_viewer_j(j, matching, rounded_weights):
	# warning: explodes if the list l is empty
	curr_min = None
	curr_idx = None
	
	for i in range(len(matching)):
		v = rounded_weights[i, j]
		
		if (curr_min is None or v < curr_min) and j in matching[i]:
			curr_min = v
			curr_idx = i
	
	return curr_idx

### The following functions are for the multiplication auction ("mu") algorithm.

# Compute a full fair matching
def mu_compute_fair_matching(weights, fairness, b):
	# b is the number of movies per viewer
	n, m = weights.shape
	matching = [[] for _ in range(n)]
	
	w_max = np.max(weights)
	scaled_weights = weights * n / (eps * w_max)
	
	# TODO k_max could change with an update...?
	k_max = ilog(w_max)
	k_min = math.ceil(-ilog(eps))
	
	rounded_weights = (1+eps)**ilog(scaled_weights)
	
	Q = [[] for i in range(n)]
	
	y = np.zeros((m,))
	
	for i in range(n):
		for j in range(m):
			for k in range(-k_min, ilog(scaled_weights[i, j])+1):
				Q[i].append((k, j))
	for i in range(n):
		# we will take the last element every time
		Q[i].sort(key=lambda x: x[0])
	
	demands = np.floor(b * fairness * n).astype(int) # fairness needs to be a numpy array
	tot_demand = sum(demands)
	
	for i in range(n):
		for m in range(b):
			print("fill", i, matching)
			# number of viewers to come after i
			viewers_left = b*(n - (i+1)) + (m-(b+1))
			
			tot_demand = mu_match(i, matching, rounded_weights, Q, y,
						      fairness, demands, viewers_left, tot_demand, b)
	
	if b == 1:
		for i in range(n):
			matching[i] = matching[i][0] if len(matching[i]) > 0 else -1
	
	# TODO match what's left
	
	# assert(tot_demand == 0 and sum(demands) <= 0)
	
	return (matching, rounded_weights, Q, y, fairness, demands, tot_demand, w_max, k_max, k_min)

# Match i
def mu_match(i, matching, rounded_weights, Q, y, fairness, demands, viewers_left, tot_demand, b):
	assert(len(matching[i]) < b)
	n, m = rounded_weights.shape
	
	while len(Q[i]) != 0:
		(k, j) = Q[i].pop()
		
		util_ij = rounded_weights[i, j] - y[j]
		if util_ij >= (1+eps)**k:
			matching[i].append(j)
			demands[j] -= 1
			if demands[j] >= 0:
				tot_demand -= 1
			
			if tot_demand > viewers_left or demands[j] == 0: # TODO investigate
				y[j] += eps * util_ij
			
			if tot_demand > viewers_left:
				# feasibility problem
				# find lightest edge to j
				lightest_viewer = lightest_viewer_j(j, matching, rounded_weights)
				lightest_viewer_lightest_movie = argmin(matching[lightest_viewer], lambda i, x: rounded_weights[lightest_viewer, x])
				
				del matching[lightest_viewer][lightest_viewer_lightest_movie]
				demands[j] += 1
				if demands[j] > 0:
					tot_demand += 1
				
#				print("rec match", lightest_viewer, len(Q[lightest_viewer]))
				print("rematch ", lightest_viewer, len(matching[lightest_viewer]), len(Q[lightest_viewer]))
				tot_demand = mu_match(lightest_viewer, matching, rounded_weights, Q, y,
						              fairness, demands, viewers_left, tot_demand, b)
				# TODO here we don't return?
				return tot_demand
			else:
				return tot_demand
	
	return tot_demand

# The update functions for the mu algorithm are now obsolete.

### To compute the optimum
	
def opt(weights, fairness, b):
	n, m = weights.shape
	
#	env = gp.Env(empty=True)
#	env.setParam('OutputFlag', 0)
#	env.start()
#	
#	model = gp.Model("fairmatch", env=env)
#	x = model.addVars(n, m, vtype=GRB.BINARY)
#	dict_weights = {(i, j): weights[i, j] for i in range(n) for j in range(m)} # to please gurobi
#	
#	model.addConstrs((gp.quicksum(x[(i, j)] for j in range(m)) <= 1 for i in range(n)))
#	model.addConstrs((gp.quicksum(x[(i, j)] for i in range(n)) >= math.floor(fairness[j] * n) for j in range(m)))
#
#	model.setObjective(x.prod(dict_weights), GRB.MAXIMIZE)
#
#	model.optimize()
	
	obj = -weights.flatten()
	# obj[i*m + j] is -weights[i, j]
	
	# m fairness constraints, n matching constraints for the viewers
	A = np.zeros((m+n, n*m))
	bv = np.zeros((m+n,))
	
	# first, the fairness constraints, then the matching constraints
	
	# fairness constraints
	for j in range(m):
		bv[j] = -np.floor(b*fairness[j]*n)
		for i in range(n):
			A[j][i*m + j] = -1
	
	# viewer-side matching constraints
	for i in range(n):
		bv[i + m] = 1
		for j in range(m):
			A[i+m][i*m + j] = b
	
	print("coefs A", A, "bv", bv)
	
	res = sopt.linprog(obj, A, bv, bounds=(0, 1))
	
	print("opt res", res.status)
	
	matching = [[] for _ in range(n)]
	for i, j in itt.product(list(range(n)), list(range(m))):
		if res.x[i*m + j] >= 0.9: # just in case the solver gets creative
			matching[i].append(j)
	
	if b == 1:
		for i in range(n):
			matching[i] = matching[i][0] if len(matching[i]) > 0 else -1

	return matching