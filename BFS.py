import numpy as np
import copy

from sys import maxsize


class Graph:
    def __init__(self, vertices):
        self._vertices = vertices
        self._adjMatrix = np.zeros((vertices, vertices))  # symmetric and full connected
        self._symPartialEdges = copy.deepcopy(self._adjMatrix)  # symmetric and partial connected
        self._asyDirectEdges = copy.deepcopy(self._adjMatrix)  # asymmetric and full connected
        self._asyPartialEdges = copy.deepcopy(self._adjMatrix)  # asymmetric and partial connected
        self.bestCost = np.inf
        self.bestPath = []

    def coordinates(self):
        rng = np.random.RandomState(10)
        return [np.array((rng.randint(-100, 100), rng.randint(-100, 100), rng.randint(0, 50))) for i in
                range(self._vertices)]

    ''' Symmetrical and full direct connected '''

    def insert_edges(self):
        coords = self.coordinates()
        for row in range(self._vertices):
            for col in range(self._vertices):
                self._adjMatrix[row][col] = np.linalg.norm(
                    coords[row] - coords[col])  # euclidean distance between cities

    ''' Asymmetrical and full direct connected'''

    def asy_direct_graph(self):
        coords = self.coordinates()
        for row in range(self._vertices):
            for col in range(self._vertices):
                if coords[row][2] > coords[col][2]:
                    self._adjMatrix[row][col] = 0.9 * (np.linalg.norm(coords[row] - coords[col]))  # euclidean distance
                elif coords[row][2] < coords[col][2]:
                    self._adjMatrix[row][col] = 1.1 * (np.linalg.norm(coords[row] - coords[col]))  # euclidean distance
                else:
                    self._adjMatrix[row][col] = np.linalg.norm(coords[row] - coords[col])  # euclidean distance

    ''' Delete 20 % of the edges --> symmetric and partially connected '''

    def sym_partial_graph(self, matrix):  # Matrix is whether asymmetric or symmetric
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

        self._adjMatrix = new_matrix20_out

    ''' Asymmetrical and partially connected'''

    def asy_partial_graph(self):
        pass

    def edge(self, s):
        edges = []
        e = s
        for edge in range(self._vertices):
            if self._adjMatrix[s][edge] != 0:
                edges.append(edge)
        return edges

    def bfs(self, s):
        i = [(s, [s])]
        while i:
            (vertix, path) = i.pop(0)
            for v in set(self.edge(vertix)) - set(path):
                if len(path + [v]) == self._vertices:
                    # yield path + [v]
                    # Update the bestcost and bestpath
                    self.best_path(path + [v])
                else:
                    i.append((v, path + [v]))

    def best_path(self, path):
        path = path + [path[0]]
        distance = 0
        for root in range(len(path) - 1):
            if self._adjMatrix[path[root]][path[root + 1]] != 0:
                distance += self._adjMatrix[path[root]][path[root + 1]]
            else:
                distance = np.inf
                break
        if distance < self.bestCost:
            self.bestCost = distance
            self.bestPath = path


net = Graph(6)
net.insert_edges()
net.bfs(3)
print(net.bestPath, net.bestCost)

