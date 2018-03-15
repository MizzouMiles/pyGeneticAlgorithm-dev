from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math, cmath, random, time, sys
import scipy.interpolate

__author__ = "Miles V. Barnhart"
__copyright__ = "Copyright 2018, Miles V. Barnhart"
__credits__ = "Miles V. Barnhart"
__license__ = "GPL"
__version__ = "5.1"
__maintainer__ = "Miles V. Barnhart"
__email__ = "MilesBarnhart@gmail.com"
__status__ = "Development"

def objective( x ): # VER. 5.1

	a = 1.0
	b = 100.0
	obj = (a - x[0])**2 + b*(x[1] - x[0]**2)**2 # Rosenbrock function where f(a, a**2) = 0
	fitness = 1/(1+obj)
	return (obj, fitness)

def init_pop( bounds , n_pop ):	

	[row, col] = np.shape(bounds)	# Shape of bounds
	pop = np.zeros((row, n_pop))	# Empty population array

	for n in xrange(0, row):		
		for i in xrange(0, n_pop):
			pop[n, i] = random.uniform(
										bounds[n, 0],
										bounds[n, 1]
									  )

	return pop # pop = n X n_pop individuals

def crossover( parents , P_c , bounds , i_gen , n_gen, fit_in):# , cross_gene):
	# Input: 
	# 		 nx2 structure with parents denoted by the columns
	# 	    
	# Output: 
	#		 nx2 structure with children denoted by columns

	[m, n] = np.shape(parents)
	children = parents

	gene_switched1 = random.randrange(0, m)
	gene_switched2 = random.randrange(0, m)

	children[gene_switched1, 0] = parents[gene_switched1, 1]
	children[gene_switched2, 1] = parents[gene_switched2, 0]

	return children

def mutation( child , bounds , ith_gen , n_gen ):
	
	m = len(child)
	zombie = np.zeros((1,m))
	R = random.uniform(0, 1)
	fG = (1 - ith_gen/n_gen)*0.05#random.uniform(0.01,0.05)
	#fG = 0.05

	mg = random.randrange(0,m)

	if R < 0.5:

		zombie[0,mg] = child[mg] + (bounds[mg,1] - child[mg])*fG
			
	else: 

		zombie[0,mg] = child[mg] - (child[mg] - bounds[mg,0])*fG

	return zombie

def mating( parents , f_min, f_avg, f_max, gen_fitness, bounds, ith_gen, n_gen):

	[m, n] = np.shape(parents)

	# Shuffle array of parents
	alpha = parents[:, -1]
	b = np.arange(0, n)
	np.random.shuffle(b)
	parents = parents[:, b]
	gen_fitness = gen_fitness[b]

	children = np.zeros((m, n))

	Pc_def = 0.95 # Default crossover probability
	Pm_def = 0.05 # Default mutation probability

	a2 = 0.5
	b2 = 1.0

	for i in xrange(0, int(n), 2):
		
		ind = random.randrange(0, n)
		ind2 = random.randrange(0, n)
		f_ = max(gen_fitness[i:i+2])
		fit_in = np.argmax(gen_fitness[i:i+2])

		if (f_avg/f_max) > a2 and (f_min/f_max) > b2: Pc = Pc_def/(1 - (f_min/f_max))

		else: Pc = Pc_def

		if (f_avg/f_max) > a2 and (f_min/f_max) > b2: Pm = Pm_def/(1 - (f_min/f_max))

		else: Pm = Pm_def
			
		if Pc >= Pc_def: 
			
			children[:,i:i+2] = crossover(parents[:,i:i+2] , gen_fitness[i:i+2], bounds , ith_gen , n_gen, fit_in)
			
		# Crossover doesn't occur	
		else: children[:,i:i+2] = parents[:, i:i+2]

		# Mutation
		if Pm >= Pm_def: 
			
			children[:,i] = mutation(children[:,i] , bounds , ith_gen , n_gen )

	return children

def fitness_eval( pop ):

	[m, n] = np.shape(pop)
	fitness = np.zeros((m+1, n)) # Pre-allocate [[population],[fitness]] structure
	fitness[:-1, :] = pop # Set top rows m to population in i_gen
	obj = (np.zeros((1, n)))# Pre-allocate objective function values

	for i in xrange(0, n):

		(obj[0,i], fitness[m, i]) = objective(pop[:,i]) # Compute fitness of each individual in the population
	
	fitness = fitness[:, np.argsort(fitness[m,:])] # Sort population by fitness (worst --> best)

	gen_fitness = fitness[m, :] # All generation fitness scores
	norm_fitness = fitness[m, :]/np.max(fitness[m, :])

	f_avg = np.mean(gen_fitness)#/np.sum(gen_fitness))#/np.sum(gen_fitness)
	
	f_max = np.max(gen_fitness)
	f_min = np.min(gen_fitness)

	sorted_pop = fitness[:-1, :] # Population sorted by fitness - fitness values removed
	obj_min = np.min(obj)
	return(sorted_pop, gen_fitness, f_min, f_avg, f_max, norm_fitness, obj_min)  # Return m X n/2 sized array w/ fittest individuals

def best_individual(opt):

	[m,n] = np.shape(opt)
	index = 0
	best_indiv = objective(opt[:,0])

	for i in xrange(1, n):

		temp_best = objective(opt[:,i])	 

		if temp_best < best_indiv:

			best_indiv = temp_best
			index = i

	return(best_indiv, index)

def GA( gen_max , n_pop , bounds , stall_lim, fit_tolerance):

	stall = 0
	yplt = np.zeros((1, gen_max))
	current_opt = []
	gen = []
	avg_fitness = []
	mean_fitness = float()
	best_individual = []
	best_fitness = float()
	max_fitness = []
	elite_n = int(10) # Number of elites to be preserved across generations
	n_rogue = int(2)
	elite_f_avg = []
	pop = init_pop(bounds, n_pop)

	print('Generation   Elite Count   Best Fitness      Mean Fitness     Min[obj]  	Gen. Stall	   Time (s)\n')

	plt.ion()

	fit_plt, (ax1, ax2) = plt.subplots(2, 1)

 	a = 1.0
	b = 100.0
	x = np.linspace(-10, 10, 100)
	y = np.linspace(-10, 10, 100)

	obj2 = [(a - x[i])**2 + b*(x[i] - y[i]**2)**2 for i in xrange(0, 100)]
	obj2 = obj2/max(obj2) # Normalize objective function (plotting)
	xi, yi = np.meshgrid(x, y)
	
	# Interpolate
	rbf = scipy.interpolate.Rbf(x, y, obj2, function='linear')
	zi = rbf(xi, yi)
	ax1.imshow(zi, vmin=min(obj2), vmax=max(obj2), origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])

 	plt.plot(a, a**2, '+r', markersize=18)

	i = np.arange(0, gen_max)

	for j in xrange(0, gen_max):

		t0 = time.time()	

		(sorted_pop, gen_fitness, f_min, f_avg, f_max, f_norm, obj_min) = fitness_eval(pop) # n X n/2 elites from pop
		elites = sorted_pop[:, int(n_pop - elite_n):]
		remaining_pop = sorted_pop[:, :int(n_pop - elite_n)]

		pop = mating(
					  sorted_pop,
					  f_min,
					  f_avg,
					  f_max,
					  gen_fitness,
					  bounds,
					  i[j],
					  gen_max
					  ) # n X n elites and offspring

		pop = np.hstack((pop[:,(elite_n):], elites))

		gen.append(i[j])

		avg_fitness.append(abs(f_avg - 1))
		max_fitness.append(abs(f_max - 1))

		t1 = time.time()
		print '%6.0f    %6.0f      \t%8.4e      \t%8.4e	    %8.8f      %6.0f       %8.4f' % \
		       (i[j], elite_n, f_max, f_avg, obj_min, stall, t1-t0)

		ax1.plot(pop[0,:], pop[1,:], '*b', elites[0,0], markersize=0.25)
		
		ax1.plot(a, a**2, '+r', markersize=18)
		ax1.plot(elites[0, 0], elites[1, 0], 'og')
		ax1.grid(True)
		ax1.set_title('Rosenbrock function: f(x,y) = (1 - x)^2 + 100*(y - x^2)^2')
		ax1.set_xlabel('X')
		ax1.set_ylabel('Y')

		ax2.semilogy(np.arange(0,j), avg_fitness[0:j],'+b')
		ax2.semilogy(np.arange(0,j), max_fitness[0:j],'ok')
		ax2.grid(True)
		ax2.set_title('Generation vs. Fitness')
		ax2.set_xlabel('Generation')
		ax2.set_ylabel('Fitness')
		ax2.legend('Best', 'Avg.')
		fit_plt.canvas.flush_events()
		time.sleep(0)

		if objective(elites) <= fit_tolerance: # Termintation of funcTol criterion met
			print('Objective function tolerance reached! Terminating computation...')
			break

		# Gen. stall limit
		if j >= gen_max/10:
			if max_fitness[i[j]] == max_fitness[i[j-1]]:
				stall += 1

			if max_fitness[i[j]] != max_fitness[i[j-1]] and stall >= 1:
				stall = 0

			if stall >= stall_lim: 
				print('Generation stall limit reached! Terminating computation...')
				break

	plt.show()
	plt.ioff()
	plt.close('all')
	
	return (
		     np.asarray(pop), 
		     np.asarray(mean_fitness), 
		     np.asarray(avg_fitness), 
		     np.asarray(max_fitness)
		     )

def run():

	NVars = 2 # No. of variables
	bounds = np.array(np.mat('-10, 10;\
							  -10,  10')) # Variable bounds

	n_pop = 100*NVars		# Population size
	gen_max = 500  # Maximum generations
	stall_lim = 100	# Generation stall limit
	fit_tolerance = 1e-6; # Objective function tolerance

	while (n_pop / 2)%2 != 0:

		print('Population Size Slightly Increased to Avoid Error!')
		n_pop += 1

	(opt,
	mean_fitness,
	avg_fit, 
	max_fit) = GA(
				   gen_max,
				   n_pop, 
				   bounds, 
				   stall_lim, 
				   fit_tolerance
				 )

	(best_indiv, index) = best_individual(opt)
	optimal_params = opt[:, index]
	
	print('\n*** Optimal Parameters ***')
	print 'x1 = %f' % optimal_params[0]
	print 'x2 = %f' % optimal_params[1]

	(obj, f) = objective(optimal_params)
	print 'objective function final value: %f' % (obj)

run()
