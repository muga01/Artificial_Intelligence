# Create the Graph
import numpy as np
import copy


class Graph:
    def __init__(self, vertices):
        self._vertices = vertices
        self._edges = np.zeros((vertices, vertices))  # symmetric and direct connected
        self._symPartialEdges = copy.deepcopy(self._edges)  # symmetric and partial connected
        self._asyDirectEdges = copy.deepcopy(self._edges)  # asymmetric and direct connected
        self._asyPartialEdges = copy.deepcopy(self._edges)  # asymmetric and partial connected

    def coordinates(self):
        rng = np.random.RandomState(10)
        x, y, z = rng.randint(-100, 100), rng.randint(-100, 100), rng.randint(0, 50)
        return [np.array((rng.randint(-100, 100), rng.randint(-100, 100), rng.randint(0, 50))) for i in
                range(self._vertices)]

    def insert_edges(self):  # edges for full connected symmetrical
        coords = self.coordinates()
        for row in range(self._vertices):
            for col in range(self._vertices):
                self._edges[row][col] = np.linalg.norm(coords[row] - coords[col])  # euclidean distance between cities

    ''' Delete 20 % of the edges --> symmetric and partially connected '''

    def sym_asy_partial_graph(self, matrix):  # Matrix is whether asymmetric or symmetric
        import copy
        delete = []
        while True:
            # choose randomly a pair of nodes to delete
            row = np.random.randint(0, self._vertices - 1)
            col = np.random.randint(0, self._vertices - 1)
            if row != col:
                delete.append((row, col))
            if len(delete) == int((self._vertices ** 2 - self._vertices) * 0.2):
                break

        # Take a deep copy of the adjascent matrix and modify the found row,col pair
        new_matrix20_out = copy.deepcopy(matrix)

        for edge in delete:
            new_matrix20_out[edge[0]][edge[1]] = 0

        self._symPartialEdges = new_matrix20_out

    ''' Asymmetrical and full direct connected'''

    def asy_direct_graph(self):
        coords = self.coordinates()
        for row in range(self._vertices):
            for col in range(self._vertices):
                if coords[row][2] > coords[col][2]:
                    self._asyDirectEdges[row][col] = 0.9 * (
                        np.linalg.norm(coords[row] - coords[col]))  # euclidean distance
                elif coords[row][2] < coords[col][2]:
                    self._asyDirectEdges[row][col] = 1.1 * (
                        np.linalg.norm(coords[row] - coords[col]))  # euclidean distance
                else:
                    self._asyDirectEdges[row][col] = np.linalg.norm(coords[row] - coords[col])  # euclidean distance

    # Draw the Graph for visualization

    def drawGraph(self):
        pass


# ACO algorithm

class Ant:
    # Initial Parameters of ACO

    np.seterr(divide='ignore')  # Ignore warnings when dividing by zero for edge quality computation 1/L

    def __init__(self, graph, num_ants, num_iter, rho, alpha, beta):
        self.maxIter = num_iter
        self.antNo = num_ants
        self.tau0 = 10 * 1 / (graph._vertices * np.mean(graph._edges))  # Initial Pheromone Concentration
        self.tau = self.tau0 * np.ones((graph._vertices, graph._vertices))  # Initial Pheromone Matrix
        self.eta = 1.0 / graph._edges  # Edge Quality
        self.rho = rho  # Evaporation Rate
        self.alpha = alpha  # Pheromone Exponential Parameter
        self.beta = beta  # Edge Quality Exponential Parameter
        self.colony = []  # Routes for all ants in a colony
        self.nodes = graph._vertices  # Vertices of the Graph
        self.edges = graph._edges  # Adjascent Matrix list
        self.colonyMatrix = np.zeros((self.maxIter, self.antNo, 2),
                                     dtype=list)  # Path, Cost for each ant in each colony
        self.queenPath = []  # The Best Path for all colonies
        self.queenCost = np.inf  # Best Cost

    # Creating a colony made of n ants

    def createColony(self):

        tour = []

        for an in range(self.antNo):
            initial_node = np.random.randint(0, self.nodes)  # Starting Node Selected Randomly
            # initial_node = 3
            tour.append(initial_node)  # Store the first node for the an'th ant

            for ne in range(1, self.nodes):  # Choose the Rest of Nodes

                current_node = tour[-1]  # Last element of the ant tour array

                p_allNodes = (self.tau[current_node][:] ** self.alpha) * (self.eta[current_node][:] ** self.beta)

                for i in tour:
                    p_allNodes[i] = 0  # Assign p_allNodes = 0 for the previously visited nodes

                p = p_allNodes / sum(p_allNodes)  # Probability for all adjascent nodes of the current nodes

                # nextNode = self.reulletWheel(p)          # Choose the Next Node

                nextNode = np.argmax(p)  # Choose the Next Node

                tour.append(nextNode)  # Add the Node to the Current ant tour

            tour.append(initial_node)  # Complete the route for an'th ant
            # if self.edges[nextNode][initial_node] != 0: # Append only if there is a path
            self.colony.append(tour)  # Add the completed ant route to the colony

            tour = []  # Empty Tour for the Next Ant

    def reulletWheel(self, p):
        pass

    def ACOMainLoop(self):

        # maxIter colonies

        for c in range(self.maxIter):

            self.colony = []

            # Create Colony
            self.createColony()

            # Create a fitness function (path + cost)
            antNum = 0  # Initialize ant number index
            for ant in self.colony:
                distance = 0
                for node in range(len(ant) - 1):
                    # Confirm if there exist a path
                    if self.edges[ant[node]][ant[node + 1]] != 0:
                        distance += self.edges[ant[node]][ant[node + 1]]
                    else:
                        # print(ant,[ant[node],ant[node+1]])
                        distance = np.inf
                        break

                # Update the Best Ant Cost and Path
                if distance < self.queenCost:
                    self.queenCost = distance
                    self.queenPath = ant

                self.colonyMatrix[c][antNum][0] = ant
                self.colonyMatrix[c][antNum][1] = distance

                antNum += 1

            # Update pheromone Matrix, tau

            for ant in range(self.antNo):  # For each ant

                for node in range(self.nodes - 1):  # For each Node

                    currentNode = self.colonyMatrix[c][ant][0][node]
                    nextNode = self.colonyMatrix[c][ant][0][node + 1]

                    self.tau[currentNode][nextNode] += 1 / self.colonyMatrix[c][ant][1]
                    self.tau[nextNode][currentNode] += 1 / self.colonyMatrix[c][ant][1]  # Symetry matrix

            # Evaporation

            self.tau = (1 - self.rho) * self.tau

            # Display Results

            # print("Iteration: {0} Cost: {1}".format(c+1,self.queenCost))

        print("Path: {0} Cost: {1}".format(self.queenPath, self.queenCost))


net = Graph(30)
net.insert_edges()
ant = Ant(net, 10, 20, 0.05, 1, 1)
ant.ACOMainLoop()
