from scipy.stats import uniform
from pyDOE import *
import numpy as np
import subprocess
import os

#Define Range of paremeter space. Here, we assume all distributions are continuous and uniform and that variables are independent from each other. We also assume
#that the transition line lies on a parabola.

#Temperature at which the transition line crosses the muB = 0 line
T0_dist = uniform(150,10)

#Critical muB
muBC_dist = uniform(220,600)

#Curvature of the transition line
kappa_dist = uniform(0.005, 0.02)

# Scalling parameters w and rho
w_dist = uniform(0.1, 8)
rho_dist = uniform(0.1, 8)

#Angle difference between Ising axes
anglediff_dist = uniform(20,160)

#Define sampler for each parameter. This is a latin hypercube sampling algorithm that takes lhs(n points, samples = m times)

T0_sampler = lhs(1, samples=1)
muBC_sampler = lhs(1, samples=1)
kappa_sampler = lhs(1, samples=1)
w_sampler = lhs(1, samples=1)
rho_sampler = lhs(1, samples=1)
anglediff_sampler = lhs(1, samples=1)ps x -o  "%p %r %c"

#Now we actually perform the sampling. The sampler gives a number on the cdf of the distribution, so we have to use the ppf to get the actual value
lhs_samples = []

for i in range(0,T0_sampler.shape[1]):
	for j in range(0,muBC_sampler.shape[1]):
		for k in range(0,kappa_sampler.shape[1]):
			for m in range(0,w_sampler.shape[1]):
				for n in range(0,rho_sampler.shape[1]):
					for h in range(0, anglediff_sampler.shape[1]):
						lhs_samples.append(np.array( [T0_dist.ppf(T0_sampler[0][i]), kappa_dist.ppf(kappa_sampler[0][k]), muBC_dist.ppf(muBC_sampler[0][j]), anglediff_dist.ppf(anglediff_sampler[0][h]), w_dist.ppf(w_sampler[0][m]), rho_dist.ppf(rho_sampler[0][n])] ))

#Convert arrays into matrix and write parameters file
print('Number of Samples: ', len(lhs_samples),'\n\n')

lhs_samples_mat = np.matrix(lhs_samples)


# print('Running sample: \n')

for i in range(0,len(lhs_samples_mat)):
	#open parameters file
    parametersfile = open('parameters.dat','w')
    #Format array 
    sample = lhs_samples_mat[i][:]
    sample = np.array(sample)
    sample = sample.flatten()
    line = ['{:.4f}'.format(x) for x in sample]
    line.insert(0,'PAR')
    line = ' '.join(line)

    print(line)

    #Here 1 is added to the index of the sample --- Remember this for later
    dirstring = '{}'.format(i+1)
    processname = './EOS '
    processname += dirstring
    #write parameters from the sample to parameters.dat
    parametersfile.write(line)
    parametersfile.close()
    #call the subprocess and run the equation of state for sample i (output from the EOS is suppressed)
    devnull = open(os.devnull, 'w')
    subprocess.call('./EOS', stdout=devnull, stderr=devnull)
    devnull.close()





