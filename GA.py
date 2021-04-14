import pandas as pd
import numpy as np
from ypstruct import structure
import random
import matplotlib.pyplot  as plt


############### FUNCTIONS ####################################################################

def fitnessFunction(population):
    machine_slots = np.empty((11, 1), dtype=list)  # Initialize machines to store busy slots
    for i in range(1, 11):
        machine_slots[i][0] = []  # Initialize as empty list
    prev_end_time = []
    job = np.zeros((50))

    start = 0

    for i in range(11):  # for ten operations
        mac = []
        for pop in population:  # for each gene
            # print(order_info[pop][i][0], end = '_')
            machine = order_info[pop][i][0]
            onTime = order_info[pop][i][1]
            # Record if the current and previous machines matches and
            # Check if the machine is not in another job
            if (mac and machine in mac) or (machine_slots[machine][0] and job[pop] < machine_slots[machine][0][-1][-1]):
                start = machine_slots[machine][0][-1][-1]
            else:
                start = job[pop]

            mac.append(machine)

            machine_slots[machine][0].append([start, start + onTime])

            job[pop] = start + onTime  # update the next start time for the next job

    return max(job)


def run(problem, params):
    # Problem Information
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    # Parameters Information
    maxit = params.maxit
    npop = params.npop
    pc = params.pc
    nc = np.round((pc * npop) / 2) * 2
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma

    # Empty Individual
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    # Best Solution Ever Found

    bestsol = empty_individual.deepcopy()
    bestsol.cost = np.inf

    # Initialize Population
    pop = empty_individual.repeat(npop)

    for i in range(npop):
        pop[i].position = random.sample(range(nvar), nvar)
        pop[i].cost = costfunc(pop[i].position)

        if pop[i].cost < bestsol.cost:
            bestsol = pop[i].deepcopy()

    # Best Cost of Iteration

    bestcost = np.empty(maxit)

    # Main Loop

    for it in range(maxit):
        popc = []  # Children Population

        for c in range(int(nc // 2)):

            # Select two Parents for crossover
            q = np.random.permutation(npop)
            p1 = pop[q[0]]
            p2 = pop[q[1]]

            # Perform Crossover
            size = len(p1.position)
            cxpoints = [random.randint(0, size), random.randint(0, size - 1)]

            c1 = pm_crossover(p1, p2, cxpoints)
            c2 = pm_crossover(p2, p1, cxpoints)
            # c1,c2 = p1.deepcopy(),p2.deepcopy()
            # c1.position = random.sample(p1.position,50)
            # c2.position = random.sample(p2.position,50)
            # print(len(list(dict.fromkeys(c1.position))))
            # print(len(list(dict.fromkeys(c2.position))))
            # print(c1.position == c2.position)
            # print()

            # Perform Mutation
            c1 = scramble_mutate(c1, mu)
            c2 = scramble_mutate(c2, mu)

            # Evaluate first offspring
            c1.cost = costfunc(c1.position)
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()

            # Evaluate second offspring
            c2.cost = costfunc(c2.position)
            if c2.cost < bestsol.cost:
                bestsol = c2.deepcopy()

            # Add the offsprings to popc
            popc.append(c1)
            popc.append(c2)

        # Merge, sort and select
        pop += popc
        sorted(pop, key=lambda x: x.cost)
        pop = pop[0:npop]

        # Store the best cost
        bestcost[it] = bestsol.cost

        # Iteration Information

        # print("Iteration {} : Best Cost {}".format(it,bestcost[it]))

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol.position
    out.globalcost = bestsol.cost
    out.bestcost = bestcost
    return out


def crossover(p1, p2):  # Partially Matched Crossover

    p1 = p1.deepcopy()
    p2 = p2.deepcopy()

    size = len(p1.position)

    c1, c2 = [None] * size, [None] * size  # Initialize children

    # Choose cross over points

    cx1 = random.randint(0, size)
    cx2 = random.randint(0, size - 1)

    if cx2 >= cx1:
        cx2 += 1
    else:
        # swap
        cx1, cx2 = cx2, cx1

    # Apply crossover between the two points
    # 1. Move the selected range from p1 to c1 and from p2 to c2
    for i in range(cx1, cx2):
        c1[i] = p1.position[i]
        c2[i] = p2.position[i]

    # 2. Check at the same position of p2 for the elements which
    # are not copied yet to c1
    i = list(set(p2.position[cx1:cx2]) - set(c1[cx1:cx2]))

    j = [c1[p2.position.index(item)] for item in i]

    # 3. Find the position of j element in p2, replace it with i, fill the c1 at the same position

    for el in range(len(j)):
        if c1[p2.position.index(j[el])] == None:
            c1[p2.position.index(j[el])] = i[el]

        else:  # If the position is occupied
            k = c1[p2.position.index(j[el])]
            c1[p2.position.index(k)] = i[el]

    # 4. Copy the remaining elements from p2 to c1

    for i in range(len(c1)):
        if c1[i] == None:
            c1[i] = p2.position[i]

    # Repeat 2 and 3 and 4 for c2, replace c1 > c2, p1 > p2

    # 2. Check at the same position of p1 for the elements which
    # are not copied yet to c2
    i = list(set(p1.position[cx1:cx2]) - set(c2[cx1:cx2]))

    j = [c2[p1.position.index(item)] for item in i]

    # 3. Find the position of j element in p1, replace it with i, fill the c2 at the same position

    for el in range(len(j)):
        if c2[p1.position.index(j[el])] == None:
            c2[p1.position.index(j[el])] = i[el]

        else:  # If the position is occupied
            k = c2[p1.position.index(j[el])]
            c2[p1.position.index(k)] = i[el]

    # 4. Copy the remaining elements from p2 to c1

    for i in range(len(c2)):
        if c2[i] == None:
            c2[i] = p1.position[i]

    p1.position, p2.position = c1, c2
    return p1, p2


def pm_crossover(p1, p2, cxpoints):  # Partially Matched Crossover

    p1 = p1.deepcopy()
    p2 = p2.deepcopy()

    size = len(p1.position)
    c1 = [0] * size

    # Initialize position of each indices in individuals
    for i in range(size):
        c1[p1.position[i]] = i

    # Choose cross over points

    cx1 = cxpoints[0]
    cx2 = cxpoints[1]

    if cx2 >= cx1:
        cx2 += 1
    else:
        # swap
        cx1, cx2 = cx2, cx1

    # Apply crossover between the two points
    for i in range(cx1, cx2):
        # hold the selected values
        temp1 = p1.position[i]
        temp2 = p2.position[i]

        # Swap the matched value
        p1.position[i], p1.position[c1[temp2]] = temp2, temp1

        # keep positions

        c1[temp1], c1[temp2] = c1[temp2], c1[temp1]

    return p1


def scramble_mutate(x, mu):
    y = x.deepcopy()
    size = int(len(y.position) * mu)
    start = random.randint(0, len(y.position))
    if start + size >= len(y.position):
        start = 0
    temp = random.sample(y.position[start:start + size], size)
    j = 0
    for i in range(start, start + size):
        y.position[i] = temp[j]
        j += 1
    return y


### Import data, job resources and time

order = pd.read_csv('.\data\GA_task.csv')
''' [[{Resource:Time}]], index of the internal(dic) list gives an operation number, the index of external list
    gives the order number. '''
order_info = \
    [[[int(list(order.iloc[j:j + 1, i])[0]), int(list(order.iloc[j:j + 1, i + 1])[0])] for j in range(1, 12, 1)] for i
     in
     range(0, 99, 2)]

# Goal, Fitness Function


''' The goal is to minimize the above defined function named, fitnessFunction()
    which takes in population list filled with 50 different jobs
'''

# Problem Definition

problem = structure()
problem.costfunc = fitnessFunction
problem.nvar = 50
problem.varmin = 0
problem.varmax = 49

# GA Parameters

params = structure()
params.maxit = 50
params.npop = 50
params.pc = 1
params.gamma = 0
params.mu = 0.3
params.sigma = 0.1

# Run GA

out = run(problem, params)

# Results

print('Best Solution: {0} Best Cost: {1} '.format(out.bestsol, out.globalcost))
plt.plot(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iteration')
plt.ylabel('Best Cost')
plt.grid(True)
plt.show()
