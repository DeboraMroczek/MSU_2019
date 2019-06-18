from scipy.stats import uniform
from pyDOE import *

#Define Range of paremeter space. Here, we assume all distributions are uniform and that variables are independent from each other. We also assume
#that the transition line lies on a parabola.

#Temperature at which the transition line crosses the muB = 0 line
T0_dist = uniform(150,10)

#Critical muB
muBC_dist = uniform(220,400)

#Curvature of the transition line
kappa_dist = uniform(0.005, 0.02)

# Scalling parameters w and rho
w_dist = uniform(0.1, 8)
rho_dist = uniform(0.1, 8)

##DRAFT WORK

design = lhs(10, samples=1)


for i in range(0,design.shape[0]):
	for j in range(0,design.shape[1]): 
		print(T0_dist.ppf(design[i][j]))