import numpy as np, copy, math, gurobipy as gp, multiprocessing as mp, os, time, datetime
import statistics as stats, random, itertools as itt
from gurobipy import GRB
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
	
	for i, x in enumerate(matching):
		v = rounded_weights[i, j]
		
		if (curr_min is None or v < curr_min) and x == j:
			curr_min = v
			curr_idx = i
	
	return curr_idx

### The following functions are for the multiplication auction ("mu") algorithm.

# Compute a full fair matching
def mu_compute_fair_matching(weights, fairness):
	n, m = weights.shape
	matching = [-1]*n
	
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
	
	demands = np.floor(fairness * n).astype(int) # fairness needs to be a numpy array
	tot_demand = sum(demands)
	
	for i in range(n):
		# number of viewers to come after i
		viewers_left = n - i - 1
		
		tot_demand = mu_match(i, matching, rounded_weights, Q, y,
						      fairness, demands, viewers_left, tot_demand)
	
	# TODO match what's left
	assert(tot_demand == 0 and sum(demands) <= 0)
	# TODO WAIT this is exactly the opposite of what I want!!!
	#assert(all((demands[i] <= np.floor(fairness[i]*n)) for i in range(m)))
	return (matching, rounded_weights, Q, y, fairness, demands, tot_demand, w_max, k_max, k_min)

# Match i
def mu_match(i, matching, rounded_weights, Q, y, fairness, demands, viewers_left, tot_demand):
	n, m = rounded_weights.shape
	
	while len(Q[i]) != 0:
		(k, j) = Q[i].pop()
		
		util_ij = rounded_weights[i, j] - y[j]
		if util_ij >= (1+eps)**k:
			matching[i] = j
			demands[j] -= 1
			if demands[j] >= 0:
				tot_demand -= 1
			
			if tot_demand > viewers_left or demands[j] == 0: # TODO investigate
				y[j] += eps * util_ij
			
			if tot_demand > viewers_left:
				# feasibility problem
				# find lightest edge to j
				lightest_viewer = lightest_viewer_j(j, matching, rounded_weights)
				
				matching[lightest_viewer] = -1
				demands[j] += 1
				if demands[j] > 0:
					tot_demand += 1
				
#				print("rec match", lightest_viewer, len(Q[lightest_viewer]))
				tot_demand = mu_match(lightest_viewer, matching, rounded_weights, Q, y,
						              fairness, demands, viewers_left, tot_demand)
				# TODO here we don't return?
				return tot_demand
			else:
				return tot_demand
	
	return tot_demand

## The following are update functions for the MU algorithm.
	
# This is not used, see next function.
#def mu_update_delete_edge(inner_state, i, j):
#	matching, rounded_weights, Q, y, fairness, demands, tot_demand, w_max, k_max, k_min = inner_state
#	
#	Q[i] = filter(lambda x: x[1] != j, Q[i])
#	
#	if matching[i] == j:
#		tot_demand = mu_match(i, matching, rounded_weights, Q, y, fairness, demands, 0, tot_demand)
#		
#	return (matching, rounded_weights, Q, y, fairness, demands, tot_demand, w_max, k_max, k_min)

# Remove all the edges from viewer i.
def mu_update_delete_vertex_edges(inner_state, i):
	""" Remark that if we call mu_update_delete_edge for all j (and fixed i) in sequence, we will
	enter the if exactly one, for j' such that i is matched to j'. Thus imagine we do this sequence of
	calls by calling with j' at the end. When we enter the if, Q[i] is empty, so mu_match returns
	without doing anything. Thus there is no work to do to remove all the edges of a vertex, just some
	bookkeeping. """
	
	matching, rounded_weights, Q, y, fairness, demands, tot_demand, w_max, k_max, k_min = inner_state
	n, m = rounded_weights.shape
	
	Q[i] = []
	match_i = matching[i]
	if match_i != -1:
		matching[i] = -1
		# TODO that's not how it works anymore
		if demands[match_i] < math.floor(fairness[match_i] * n):
			demands[match_i] += 1
			tot_demand += 1
		
	return (matching, rounded_weights, Q, y, fairness, demands, tot_demand, w_max, k_max, k_min)

# Add a viewer i and all its weights, and update.
def mu_update_add_vertex(inner_state, i, viewer_weights):
	matching, rounded_weights, Q, y, fairness, demands, tot_demand, w_max, k_max, k_min = inner_state
	n, m = rounded_weights.shape
	
	rounded_weights[i, :] = (1+eps)**ilog(viewer_weights)
	
	Q[i] = []
	
	for j in range(m):
		for k in range(-k_min, ilog(rounded_weights[i, j]) + 1):
			Q[i].append((k, j))
	
	Q[i].sort(key=lambda x: x[0])
	
#	print("let's match!", i)
	tot_demand = mu_match(i, matching, rounded_weights, Q, y, fairness, demands, 0, tot_demand)
	
#	print(i, "matched to", matching[i])
	
	return (matching, rounded_weights, Q, y, fairness, demands, tot_demand, w_max, k_max, k_min)

# Update the matching.
def mu_update_fair_matching(inner_state, i, viewer_weights):
	# when we update, viewers_left == 0
	
	# Here, we can copy the inner_state if we want to have no correlation
	inner_state = mu_update_delete_vertex_edges(inner_state, i)
#	print("upd_fm_edges", inner_state[0])
	inner_state = mu_update_add_vertex(inner_state, i, viewer_weights)
	
	return inner_state[0], inner_state # the matching and the inner state


### To compute the optimum
	
def opt(weights, fairness):
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
	A = np.zeros((n*m, m+n))
	b = np.zeros((m+n,))
	
	# first, the fairness constraints, then the matching constraints
	
	# fairness constraints
	for j in range(m):
		b[j] = -np.floor(fairness[j]*n)
		for i in range(n):
			A[i*m + j][j] = -1
	
	# viewer-side matching constraints
	for i in range(n):
		b[i + m] = 1
		for j in range(m):
			A[i*m + j][i + m] = 1
	
	res = sopt.linprog(obj, A, b, bounds=(0, 1))
	
	matching = [-1]*n
	for i, j in itt.product(list(range(n)), list(range(m))):
		if res.x[i*m + j] >= 0.9: # just in case the solver gets creative
			matching[i] = j

	return matching