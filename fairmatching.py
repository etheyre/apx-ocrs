#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, copy, math

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
	
	demands = np.floor(fairness * n) # fairness needs to be a numpy array
	tot_demand = sum(demands)
	
	for i in range(n):
		# number of viewers to come after i
		viewers_left = n - i
		# invariant: tot_demand = sum(demands)
		tot_demand = mu_match(i, matching, rounded_weights, Q, y,
						      fairness, demands, viewers_left, tot_demand)
	
	# match what's left
	
	return (matching, rounded_weights, Q, y, fairness, demands, tot_demand, w_max, k_max, k_min)

# Match i
def mu_match(i, matching, rounded_weights, Q, y, fairness, demands, viewers_left, tot_demand):
	while len(Q[i]) != 0:
		(k, j) = Q[i].pop()
		
		util_ij = rounded_weights[i, j] - y[j]
		
		print("hey", rounded_weights[i, j], y[j])
		
		if util_ij >= (1+eps)**k:
			y[j] += eps * util_ij
			
			demand_contribution = 1 if demands[j] > 0 else 0
			matching[i] = j
			demands[j] -= demand_contribution
			tot_demand -= demand_contribution
			
			if tot_demand > viewers_left:
				# feasibility problem
				# find lightest edge to j
				lightest_viewer = argmin(matching, lambda a, b: rounded_weights[a, b])
				
				j_lightest = matching[lightest_viewer]
				if demands[j_lightest] < fairness[j_lightest]:
					demands[j_lightest] += 1
					tot_demand += 1
					
				matching[lightest_viewer] = -1
				tot_demand = mu_match(lightest_viewer, matching, rounded_weights, Q, y,
						              fairness, demands, viewers_left, tot_demand)
				# TODO here we don't return?
			else:
				return tot_demand
	return tot_demand

## The following are update functions for the MU algorithm.
	
# This is not used, see next function.
def mu_update_delete_edge(inner_state, i, j):
	matching, rounded_weights, Q, y, fairness, demands, tot_demand, w_max, k_max, k_min = inner_state
	
	Q[i] = filter(lambda x: x[1] != j, Q[i])
	
	if matching[i] == j:
		tot_demand = mu_match(i, matching, rounded_weights, Q, y, fairness, demands, 0, tot_demand)
		
	return (matching, rounded_weights, Q, y, fairness, demands, tot_demand, w_max, k_max, k_min)

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
		if demands[match_i] < fairness[match_i] * n:
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
	
	tot_demand = mu_match(i, matching, rounded_weights, Q, y, fairness, demands, 0, tot_demand)
	
	print(i, "matched to", matching[i])
	
	return (matching, rounded_weights, Q, y, fairness, demands, tot_demand, w_max, k_max, k_min)

# Update the matching.
def mu_update_fair_matching(inner_state, i, viewer_weights):
	# when we update, viewers_left == 0
	
	# Here, we can copy the inner_state if we want to have no correlation
	inner_state = mu_update_delete_vertex_edges(inner_state, i)
	print("upd_fm_edges", inner_state[0])
	inner_state = mu_update_add_vertex(inner_state, i, viewer_weights)
	
	return inner_state[0], inner_state # the matching and the inner state










### The following functions are for the rest of the algo

def ocrs(sampled_weights, online_weights, fairness, init_inner_sol, update_inner_sol):
	# The inner_state is used by the embedded algorithm. In this function, it is considered a blackbox.
	inner_state = init_inner_sol(sampled_weights, fairness)
	n, m = online_weights.shape
	
	ocrs_matching = [-1]*n
	
	for i in range(n):
		print("---- update", i)
		viewer_matching, inner_state = update_inner_sol(inner_state, i, online_weights[i, :])
		ocrs_matching[i] = viewer_matching[i]
	
	return ocrs_matching

def run(distrib, fairness):
	sampled_weights = distrib()
	online_weights = distrib()
	matching = ocrs(sampled_weights, online_weights, fairness, mu_compute_fair_matching, mu_update_fair_matching)
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
	
	demand_deficit = int(sum(filter(lambda x: x > 0, demands)))
	
	return (matching_weight, demand_deficit)

def unif_distrib(n, m):
	rng = np.random.default_rng()
	weights = np.zeros((n, m))
	for i in range(n):
		for j in range(m):
			weights[i, j] = 50-25*j+1
	
	return weights

def test():
	fairness = np.array([0.1, 0.1, 0.1])
	weights = unif_distrib(10, 3)
	m, weights = run(lambda: unif_distrib(10, 3), fairness)
	print("now, opt -----------")
	m_alg = mu_compute_fair_matching(weights, fairness)[0]
	print(m, score_matching(m, fairness, weights))
	print(m_alg, score_matching(m_alg, fairness, weights))

test()