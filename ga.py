import numpy as np
from ypstruct import structure


def run(problem, params):
    # Problem information
    costfun = problem.costfun
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    # Parameters
    maxit = params.maxit
    npop = params.npop
    pc = params.pc
    nc = np.round(pc * npop / 2) * 2
    gama = params.gama

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    bestol = empty_individual.deepcopy()
    bestol.cost = np.inf

    # Initialise population
    pop = empty_individual.repeat(npop)
    for i in range(0, npop):
        pop[i].position = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = costfun(pop[i].position)
        if pop[i].cost < bestol.cost:
            bestol = pop[i].deepcopy()

    # Best cost of interations
    bestcost = np.empty(maxit)

    # Main loop
    for it in range(maxit):
        popc = []
        for k in range(int(nc) // 2):
            # Select parents
            q = np.random.permutation(npop)
            p1 = pop[q[0]]
            p2 = pop[q[1]]

            # Perform crosssover
            c1, c2 = crosssover(p1, p2, gama)
    # Output
    out = structure()
    out.pop = pop

    return out


def crosssover(p1, p2, gama=0.1):
    c1 = p1.deepcopy()
    c2 = p2.deepcopy()
    alpha = np.random.uniform(-gama, 1 + gama, *c1.position.shape)
    c1.position = alpha * p1.position + (1 - alpha) * p2.postion
    c2.position = alpha * p2.position + (1 - alpha) * p1.postion
    return c1, c2
