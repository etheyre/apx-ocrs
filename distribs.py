import numpy as np
import random

def avg(l):
	return sum(l)/len(l)

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
	# (rarely have we seen something so arbitrary)
	k = 10
	rng = np.random.default_rng()
	weights = np.zeros((n, m))
	for i in range(n):
		l = [random.randint(0, 10) for i in range(k)]
		for j in range(m):
			weights[i, j] = rng.uniform(l[i%k]/k, (l[i%k]+1)/10)
	
	return weights


# This is the MovieLens stuff. It's not used for now. Also, it probably doesn't work.


ml_ratings = []
ml_n_users = 0
ml_m_movies = 0

def load_movielens_distrib():
	global ml_ratings, ml_n_users, ml_m_movies
	ml_ratings = []
	ml_n_users = 0
	ml_m_movies = 0
	# userId,movieId,rating,timestamp
	with open("ml-25m/ratings.csv", "r") as f:
		csv = [x.split(",") for x in f.read().split("\n")[1:]]
	
	ml_n_users = max((int(x[0]) for x in csv))
	ml_ratings = [{} for _ in range(ml_n_users)]
	
	for l in csv:
		user = int(l[0])
		movie = int(l[1])
		ml_m_movies = max(ml_m_movies, movie)
		rating = float(l[2])/5.
		ml_ratings[user][movie] = rating
	
	return ml_ratings

ml_genre_profiles = []
ml_movie_tags = []

def generate_genre_profiles():
	global ml_genre_profiles, ml_movie_tags
	ml_genre_profiles = []
	ml_movie_tags = []
	
	with open("ml-25m/movies.csv", "r") as f:
		csv = [x.split(",") for x in f.read().split("\n")[1:]]
	
	ml_movie_tags = [None for _ in range(ml_m_movies)]
	
	all_tags = set()
	
	for l in csv:
		movie = int(l[0])
		tags = set(l[2].split("|"))
		all_tags = all_tags.union(tags)
		ml_movie_tags[movie] = tags
	
	for u in range(ml_n_users):
		for t in all_tags:
			 ml_genre_profiles[u][t] = []
		 
		for m in range(ml_m_movies):
			if t in ml_movie_tags[m]:
				ml_genre_profiles[u][t].append(ml_ratings[u][m])
		
		for t in all_tags:
			if len(ml_genre_profiles[u][t]) != 0:
				ml_genre_profiles[u][t] = avg(ml_genre_profiles[u][t])
			else:
				ml_genre_profiles[u][t] = 0.2 # arbitrary

def real_movielens_distrib(n, m):
	# this disregards n and m
	global ml_ratings, ml_m_movies, ml_n_users
	
	ratings = np.zeros((ml_n_users, ml_m_movies))
	
	for u in ml_ratings:
		for m in ml_m_movies:
			ratings[u, m] = ml_ratings[u][m]
	
	return ratings

ml_filled_n = -1
ml_filled_m = -1
ml_filled_distrib = []

# yeah but for some viewers, that might be completely empty, if we only use the first m movies

def make_filled_movielens_distrib(n, m):
	# uses only the n first viewers and m first movies
	global ml_ratings, ml_m_movies, ml_n_users, ml_filled_distrib, ml_filled_n, ml_filled_m
	
	if ml_filled_n == n and ml_filled_m == m and ml_filled_distrib != []:
		return ml_filled_distrib
	
	assert(n <= ml_n_users and m <= ml_m_movies)
	
	ratings = np.zeros((n, m))
	
	for u in range(n):
		for mv in range(m):
			ratings[u, mv] = avg((ml_genre_profiles[u][t] for t in ml_movie_tags[mv]))
	
	ml_filled_distrib = ratings
	ml_filled_n = n
	ml_filled_m = m
	
	return ratings

def sample_movie_lens_distrib(n, m):
	# uses only the n first viewers and m first movies
	global ml_ratings, ml_m_movies, ml_n_users, ml_filled_distrib, ml_filled_n, ml_filled_m
	
	