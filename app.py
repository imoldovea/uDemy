import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga

#Sphere test function
def sphere(x):
    return sum(x**2)

#Probelm definition
problem = structure()
problem.costfun = sphere
problem.nvar = 5
problem.varmin = -10
problem.varmax = 10


#GA parameters
parms = structure()
parms.maxit = 100
parms.npop = 20

#Run GA
out = ga.run()problem,parms)

#Results