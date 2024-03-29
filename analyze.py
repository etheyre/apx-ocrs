import matplotlib.pyplot as plt
import sys


def analyze_log(name):
	with open(name, "r") as f:
		txt = f.read()
	
	data = []

	for l in txt.split("\n"):
		if l.startswith("res"):
			data.append(eval(l[3:]))
	
	print(len(data))
	
	x = [x[0]*x[1] for x in data] # instance size
	y = [x[2]/x[3] for x in data] # duration ratio
	plt.xscale("log")
	plt.plot(x, y, "o")
	plt.savefig("plot_" + ".".join(name.split(".")[:-1]))

analyze_log(sys.argv[1])