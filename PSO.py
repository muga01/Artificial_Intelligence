import numpy as np
import random
import matplotlib.pyplot as plt


# PROBLEM DEFINITION

# Objective cost minimization function

def function(x):
    return (1.5 - x[0] - x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * (x[1]) ** 2) ** 2 + (
                2.625 - x[0] + x[0] * (x[1]) ** 3) ** 2


nVar = 2  # Number of variables
bound = [-4.5, 4.5]  # Lower and upper bound of variables


def apply_bound(position):  # Make sure particle positions are in range
    if not -4.5 <= position[0] <= 4.5:
        position[0] = random.uniform(-4.5, 4.5)

    if not -4.5 <= position[1] <= 4.5:
        position[1] = random.uniform(-4.5, 4.5)

    return np.array(position)


# PARAMETERS OF PSO

maxIter = 100  # Maximum number of iterations
npop = 50  # Swarm size
w = 1  # Inertia Coefficient
wdamp = 0.95  # Damping ratio of Inertia Coefficient
c1 = 2  # Personal Acceleration Coefficient
c2 = 2  # Social Acceleration Coefficient

# INITIALIZATION


# Column definition: 0 --> particle position, 1 --> Particle velocity, 2 --> particle cost, 
# 3 --> particle best position, 4 --> particle best cost

particle = np.zeros((npop, 5), dtype=list)  # Create a particle population array

# Initialize members of the population

# Initialize the global best
globalBestCost = np.inf

for member in range(npop):

    # Generate a Random solution
    solution = []  # Create a particle position list
    for i in range(nVar):
        solution.append(random.uniform(-4.5, 4.5))  # Assign a random position
    particle[member][0] = np.array(solution)  # Store the solution into an array

    # Initialize Velocity to a zero array
    particle[member][1] = np.array([0, 0])

    # Evaluation of Particle Cost
    particle[member][2] = function(particle[member][0])

    # Update the Personal Best Position
    particle[member][3] = particle[member][0]

    # Update the personal Best Cost
    particle[member][4] = particle[member][2]

    # Update the Global best Position and Cost
    if particle[member][4] < globalBestCost:
        globalBestCost = particle[member][4]
        globalBestPosition = particle[member][3]

# Array to hold Best Cost Value for each Iteration
bestCosts = np.zeros(maxIter)

# MAIN LOOP OF PSO

for it in range(maxIter):

    for member in range(npop):

        # Update Particle Velocity

        particle[member][1] = (w * particle[member][1]
                               + c1 * random.uniform(0, 1) * (particle[member][3] - particle[member][0])
                               + c2 * random.uniform(0, 1) * (globalBestPosition - particle[member][0]))

        # Update Current Particle Position

        particle[member][0] = particle[member][0] + particle[member][1]
        particle[member][0] = apply_bound(particle[member][0])  # Apply Bounds

        # Compute the cost of the particle at the new position

        particle[member][2] = function(particle[member][0])

        # Update the Best Cost and Position

        if particle[member][2] < particle[member][4]:
            particle[member][4] = particle[member][2]
            particle[member][3] = particle[member][0]

            # Update the Global best Position and Cost
            if particle[member][4] < globalBestCost:
                globalBestCost = particle[member][4]
                globalBestPosition = particle[member][3]

    # Store the Best Cost value
    bestCosts[it] = globalBestCost

    # Display the iterations

    # print("Iteration: ",it+1, end=' ')
    # print("Best Minimum =", bestCosts[it])

    # Update w

    w = w * wdamp

print(globalBestPosition, globalBestCost)

# RESULTS
figure = plt.figure(figsize=(10, 6))
plt.plot(bestCosts)
plt.xlabel('Iterations')
plt.ylabel('Best Minimum')
plt.grid('on')
plt.show()

