#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, copy, math, gurobipy as gp, multiprocessing as mp, os, time, datetime
from gurobipy import GRB

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
	assert(all((demands[i] <= np.floor(fairness[i]*n)) for i in range(m)))
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
				
				print(i, lightest_viewer, demands[j], tot_demand, viewers_left)
				
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
	
	env = gp.Env(empty=True)
	env.setParam('OutputFlag', 0)
	env.start()
	
	model = gp.Model("fairmatch", env=env)
	x = model.addVars(n, m, vtype=GRB.BINARY)
	dict_weights = {(i, j): weights[i, j] for i in range(n) for j in range(m)} # to please gurobi
	
	model.addConstrs((gp.quicksum(x[(i, j)] for j in range(m)) <= 1 for i in range(n)))
	model.addConstrs((gp.quicksum(x[(i, j)] for i in range(n)) >= math.floor(fairness[j] * n) for j in range(m)))

	model.setObjective(x.prod(dict_weights), GRB.MAXIMIZE)

	model.optimize()
	
	matching = [-1]*n
	for i, j in x.keys():
		if x[i, j].x == 1:
			matching[i] = j
	
	return matching

### The following functions are for the rest of the algo

def ocrs(sampled_weights, online_weights, fairness, init_inner_sol, update_inner_sol):
	# The inner_state is used by the embedded algorithm. In this function, it is considered a blackbox.
	inner_state = init_inner_sol(sampled_weights, fairness)
	n, m = online_weights.shape
	
	ocrs_matching = [-1]*n
	
	for i in range(n):
		#print("---- update", i)
		viewer_matching, inner_state = update_inner_sol(inner_state, i, online_weights[i, :])
		ocrs_matching[i] = viewer_matching[i]
	
	return ocrs_matching

def run_dyn_alg(distrib, fairness, online_weights=None):
	sampled_weights = distrib()
	online_weights = distrib() if online_weights is None else online_weights
	matching = ocrs(sampled_weights, online_weights, fairness,
				    mu_compute_fair_matching, mu_update_fair_matching)
	return matching, online_weights

def mu_recompute_fair_matching(inner_state, i, weights_i):
	weights = inner_state[1]
	fairness = inner_state[4]
	here_weights = np.copy(weights)
	here_weights[i, :] = weights_i
	out_state = mu_compute_fair_matching(here_weights, fairness)
	return out_state[0], out_state

def run_off_alg(distrib, fairness, online_weights=None):
	sampled_weights = distrib()
	online_weights = distrib() if online_weights is None else online_weights
	matching = ocrs(sampled_weights, online_weights, fairness,
				    mu_compute_fair_matching, mu_recompute_fair_matching)
	return matching, online_weights

def opt_precompute(weights, fairness):
	return (weights, fairness)

def opt_recompute(inner_state, i, weights_i):
	weights = inner_state[0]
	fairness = inner_state[1]
	here_weights = np.copy(weights)
	here_weights[i, :] = weights_i
	matching = opt(here_weights, fairness)
	return matching, (weights, fairness)

def run_ocrs_opt(distrib, fairness, online_weights=None):
	sampled_weights = distrib()
	online_weights = distrib() if online_weights is None else online_weights
	matching = ocrs(sampled_weights, online_weights, fairness,
				    opt_precompute, opt_recompute)
	return matching, online_weights

# Output the weight collected by the matching, and the total deficit in fairness.
def score_matching(matching, fairness, weights):
	n, m = weights.shape
	
	matching_weight = 0
	demands = np.floor(fairness * n)
	
	for i in range(n):
		j = matching[i]
		if j == -1:
			continue
		matching_weight += weights[i, j]
		demands[j] -= 1
	
	# demands is fairness - loads
	
	demand_deficit = int(sum(filter(lambda x: x > 0, demands)))
#	demands = np.array(list(map(lambda x: max(0, x), demands)))
	return (matching_weight, demand_deficit, demands)

def unif_distrib(n, m):
	rng = np.random.default_rng()
	weights = np.zeros((n, m))
	for i in range(n):
		for j in range(m):
			weights[i, j] = rng.random()*3+1
			assert(weights[i, j] >= 0)
	
	return weights

def fairness_ocrs_mu():
	n = 100
	m = 10
	N = 1
	fairness = np.array([0.095]*m)

	tot_demands = np.zeros((m,), int)
	for i in range(N):
		print(i)
		m_ocrs, weights = run_off_alg(lambda: unif_distrib(n, m), fairness)
		s_ocrs, fair_ocrs, final_demands_ocrs = score_matching(m_ocrs, fairness, weights)
		tot_demands += final_demands_ocrs.astype(int)
	
	print(tot_demands/N) # here, negative is good

def run_analyze_ocrs_mu(args):
	n, m, fairness = args
	m_ocrs, weights = run_off_alg(lambda: unif_distrib(n, m), fairness)
	s_ocrs, fair_ocrs, final_demands_ocrs = score_matching(m_ocrs, fairness, weights)
	print(fair_ocrs, final_demands_ocrs)
	return (s_ocrs, fair_ocrs, final_demands_ocrs.astype(int))

def fairness_ocrs_mu_parallel():
	n = 100
	m = 10
	N = 100
	fairness = np.array([0.095]*m)
	
	print("starting on", os.cpu_count(), "glorious CPUs")
	t = time.time()
	with mp.Pool() as p:
		res = p.map(run_analyze_ocrs_mu, [(n, m, fairness)]*N)
	
	print("took", str(datetime.timedelta(seconds=time.time()-t)))
	
	tot_demands = sum([x[2] for x in res]).astype(float)/N
	
	print(tot_demands)
	
def fairness_ocrs_opt():
	fairness = np.array([0.3, 0.4, 0.1])
	n = 10
	m = 3
	N = 100
	tot_demands = np.zeros((m,), int)
	for i in range(N):
		m_ocrs, weights = run_ocrs_opt(lambda: unif_distrib(n, m), fairness)
		s_ocrs, fair_ocrs, final_demands_ocrs = score_matching(m_ocrs, fairness, weights)
		tot_demands += final_demands_ocrs.astype(int)
	
	print(tot_demands/N)

def test():
	fairness = np.array([0.3, 0.4, 0.1])
	weights = unif_distrib(10, 3)
#	m, weights = run_dyn_alg(lambda: unif_distrib(10, 3), fairness)
#	print("online-apx", m, score_matching(m, fairness, weights))
	m_ocrs_apx, _ = run_off_alg(lambda: unif_distrib(10, 3), fairness, online_weights=weights)
	m_ocrs_opt, _ = run_ocrs_opt(lambda: unif_distrib(10, 3), fairness, online_weights=weights)
	m_alg = mu_compute_fair_matching(weights, fairness)[0]
	m_opt = opt(weights, fairness)
	
	s_ocrs_apx, fair_ocrs_apx, _ = score_matching(m_ocrs_apx, fairness, weights)
	s_ocrs_opt, fair_ocrs_opt, _ = score_matching(m_ocrs_opt, fairness, weights)
	s_alg, fair_alg, _ = score_matching(m_alg, fairness, weights)
	s_opt, fair_opt, _ = score_matching(m_opt, fairness, weights)
	
	print("ocrs-opt", m_ocrs_opt, s_ocrs_opt, fair_ocrs_opt)
	print("ocrs-apx", m_ocrs_apx, s_ocrs_apx, fair_ocrs_apx)
	print("offline-apx", m_alg, s_alg, fair_alg)
	print("opt", m_opt, s_opt, fair_opt)
	assert(fair_alg == 0 and fair_opt == 0)
	assert(s_alg <= (1+eps) * s_opt)

#test()
fairness_ocrs_mu_parallel()
#fairness_ocrs_mu()
#fairness_ocrs_opt()