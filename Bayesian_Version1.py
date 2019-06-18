from scipy.stats import uniform
from pyDOE import *
import numpy as np

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
anglediff_sampler = lhs(1, samples=1)

#Now we actually perform the sampling. The sampler gives a number on the cdf of the distribution, so we have to use the ppf to get the actual value
lhs_samples = []

for i in range(0,T0_sampler.shape[1]):
	for j in range(0,muBC_sampler.shape[1]):
		for k in range(0,kappa_sampler.shape[1]):
			for m in range(0,w_sampler.shape[1]):
				for n in range(0,rho_sampler.shape[1]):
					lhs_samples.append(np.array([T0_dist.ppf(T0_sampler[0][i]),muBC_dist.ppf(muBC_sampler[0][j]),kappa_dist.ppf(kappa_sampler[0][k]), w_dist.ppf(w_sampler[0][m]),rho_dist.ppf(rho_sampler[0][n]),  w_dist.ppf(w_sampler[0][m]),anglediff_dist.ppf(anglediff_sampler[0][h])]))

#Convert arrays into matrix and write parameters file

lhs_samples_mat = np.matrix(lhs_samples)

with open('parameters.dat','wb') as parametersfile:
    for sample in lhs_samples_mat:
        np.savetxt(parametersfile, sample, fmt='%.2f')
