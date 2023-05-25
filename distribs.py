import numpy as np
import random

def unif_distrib(n, m):
	rng = np.random.default_rng()
	weights = np.zeros((n, m))
	for i in range(n):
		for j in range(m):
			weights[i, j] = rng.random()*3+1
			assert(weights[i, j] >= 0)
	
	return weights

def liked_distrib(n, m):
	# make ten movie categories (j%10) and assign a number l in [0, 9] to each
	# then, draw the weights in random(l/10, (l+1)/10) for each category
	# really have we seen something so arbitrary
	k = 10
	rng = np.random.default_rng()
	weights = np.zeros((n, m))
	for i in range(n):
		l = [random.randint(0, 10) for i in range(k)]
		for j in range(m):
			weights[i, j] = rng.uniform(l[i%k]/k, (l[i%k]+1)/10)
	
	return weights