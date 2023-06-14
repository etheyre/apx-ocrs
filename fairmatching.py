#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, copy, math, multiprocessing as mp, os, time, datetime
import statistics as stats, random, itertools as itt

from distribs import *
from algos import *

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

def ocrs(sampled_weights, online_weights, fairness, init_inner_sol, update_inner_sol):
	# The inner_state is used by the embedded algorithm. In this function, it is considered a blackbox.
	inner_state = init_inner_sol(sampled_weights, fairness)
	n, m = online_weights.shape
	
	ocrs_matching = [-1]*n
	
	for i in range(n):
		viewer_matching, inner_state = update_inner_sol(inner_state, i, online_weights[i, :])
		ocrs_matching[i] = viewer_matching[i]
	
	return ocrs_matching







def run_dyn_alg(distrib, fairness, online_weights=None):
	sampled_weights = distrib()
	online_weights = distrib() if online_weights is None else online_weights
	matching = ocrs(sampled_weights, online_weights, fairness,
				    mu_compute_fair_matching, mu_update_fair_matching)
	return matching, online_weights

def mu_recompute_fair_matching(inner_state, i, weights_i, b):
	weights = inner_state[1]
	fairness = inner_state[4]
	here_weights = np.copy(weights)
	here_weights[i, :] = weights_i
	out_state = mu_compute_fair_matching(here_weights, fairness)
	return out_state[0], out_state

def run_off_alg(distrib, fairness, b, online_weights=None):
	sampled_weights = distrib()
	online_weights = distrib() if online_weights is None else online_weights
	matching = ocrs(sampled_weights, online_weights, fairness,
				    lambda w, f: mu_compute_fair_matching(w, f, b), lambda w, f: mu_recompute_fair_matching(w, f, b))
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

def stupidest_precompute(weights, fairness):
	return opt(weights, fairness)

def stupidest_recompute(inner_state, i, weights_i):
	return inner_state, inner_state

def run_ocrs_stupidest_algo(distrib, fairness, online_weights=None):
	sampled_weights = distrib()
	online_weights = distrib() if online_weights is None else online_weights
	matching = ocrs(sampled_weights, online_weights, fairness,
				    stupidest_precompute, stupidest_recompute)
	return matching, online_weights




def run_analyze_ocrs_mu(args):
	n, m, fairness, b = args
	start = time.time()
	m_ocrs, weights = run_off_alg(lambda: unif_distrib(n, m), fairness, b)
	m_opt = opt(weights, fairness, b)
	s_opt = 1
	s_opt, _, _ = score_matching(m_opt, fairness, weights)
	s_ocrs, fair_ocrs, final_demands_ocrs = score_matching(m_ocrs, fairness, weights)
	return (s_ocrs, s_ocrs/s_opt, fair_ocrs, list(final_demands_ocrs), time.time()-start)

def run_analyze_ocrs_opt(args):
	n, m, fairness = args
	start = time.time()
	m_ocrs, weights = run_ocrs_opt(lambda: unif_distrib(n, m), fairness)
	#m_opt = opt(weights, fairness)
	s_opt = 1
	#s_opt, _, _ = score_matching(m_opt, fairness, weights)
	s_ocrs, fair_ocrs, final_demands_ocrs = score_matching(m_ocrs, fairness, weights)
	return (s_ocrs, s_ocrs/s_opt, fair_ocrs, list(final_demands_ocrs), time.time()-start)

def run_analyze_stupidest(args):
	n, m, fairness = args
	m_ocrs, weights = run_ocrs_stupidest_algo(lambda: liked_distrib(n, m), fairness)
	m_opt = opt(weights, fairness)
	s_opt, _, _ = score_matching(m_opt, fairness, weights)
	s_ocrs, fair_ocrs, final_demands_ocrs = score_matching(m_ocrs, fairness, weights)
	return (s_ocrs, s_opt)



def run_parallel(f, fairness, n, m, b, N, cpus=-1):
	print("starting on", os.cpu_count(), "glorious CPUs")
	t = time.time()
	if cpus == -1:
		cpus = os.cpu_count()
	with mp.Pool() as p:
		res = p.map(f, [(n, m, fairness, b)]*N)
	
	print("took", str(datetime.timedelta(seconds=time.time()-t)))
	
	tot_demands = sum([np.array(x[3]) for x in res])/N
	ratios = [x[1] for x in res]
	print(min(ratios), max(ratios), sum(ratios)/N, stats.variance(ratios), stats.quantiles(ratios))
	print(tot_demands)
	return [x[-1] for x in res], ratios, list(tot_demands)

def fairness_ocrs_mu_parallel(fairness, n=100, m=10, b=1, N=1000):	
	return run_parallel(run_analyze_ocrs_mu, fairness, n, m, b, N, -1)
	
def fairness_ocrs_opt_parallel(fairness, n=100, m=10, N=1000):
	return run_parallel(run_analyze_ocrs_opt, fairness, n, m, N, -1)

def fairness_blind_parallel(fairness, n=100, m=10, N=1000):
	return run_parallel(run_analyze_stupidest, fairness, n, m, N, -1)

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

def compare_running_times():
	N = 50
	time_data = []
	params = [(n, m, np.array([1/(m+1)]*m)) for (n, m) in itt.product([10, 50, 1000, 10000], [1, 10, 20, 100, 500])]
	for (n, m, fairness) in params:
		print("start", n, m, 1/(m+1))
		times_mu, ratios_mu, avg_leftover_mu = fairness_ocrs_mu_parallel(fairness, n, m, N)
		times_opt, ratios_opt, avg_leftover_opt = fairness_ocrs_opt_parallel(fairness, n, m, N)
		
		avg_time_mu = avg(times_mu)
		avg_ratio_mu = avg(ratios_mu)
		avg_time_opt = avg(times_opt)
		avg_ratio_opt = avg(ratios_opt)
		time_data.append((n, m, avg_time_mu, avg_time_opt, avg_ratio_mu, avg_ratio_opt, avg_leftover_mu, avg_leftover_opt))
		print("res", (n, m, avg_time_mu, avg_time_opt, avg_ratio_mu, avg_ratio_opt, avg_leftover_mu, avg_leftover_opt))
	
	with open("times.dat", "w") as f:
		f.write(str(time_data))

n = 10
m = 2
b = 3
N = 20
fairness = np.array([1/(m+1)]*m)
times_mu, ratios_mu, avg_leftover_mu = fairness_ocrs_mu_parallel(fairness, n, m, b, N)
print(times_mu, ratios_mu, avg_leftover_mu)

#compare_running_times()
#test()
#fairness_ocrs_mu_parallel()
#run_stupidest_parallel()
#fairness_ocrs_mu()
#fairness_ocrs_opt()