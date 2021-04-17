import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga


# Sphere Test Cost Function
def sphere(x):
    return sum(x ** 2)


# Probelm definition
problem = structure()
problem.costfun = sphere
problem.nvar = 5
problem.varmin = -10
problem.varmax = 10

# GA parameters
params = structure()
params.maxit = 100
params.npop = 20
params.pc = 1
params.gama = 0.1

# Run GA
out = ga.run(problem, params)


# Results
