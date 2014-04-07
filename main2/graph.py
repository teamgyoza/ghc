#from math import sqrt
from cmath import phase

a = 114180.046772
b = 111803.550422

class Edge(object):

    __slots__ = ('idx', 'distance', 'cost', 'original_distance','start', 'stop', 'cars', 'efficiency', 'reverse', 'visits', 'angle')
    def __init__(self, idx, start, stop, distance, cost, ):
        self.idx = idx
        self.distance = distance
        self.cost = cost
        self.original_distance = distance
        self.start = start
        self.stop = stop
        self.reverse = None
        self.visits = 0
        self.cars = set()
        self.angle = phase(self.stop.position - self.start.position)
        self.efficiency = 1


    def reset(self,):
        self.visits = 0
        self.distance = self.original_distance
        self.efficiency = 1
        self.cars = set()

    def visit(self, car):
        self.cars.add(car)
        edges = [ self ]
        if self.reverse:
            edges.append(self.reverse)
        for edge in edges:
            edge.efficiency = 0
            edge.visits += 1
            edge.distance = 0

    def __repr__(self,):
        return "Edge(%i, %i)" %(self.start.idx, self.stop.idx)

class Node(object):

    __slots__ = ('idx', 'neighbors', 'x', 'y', 'edges', 'position', 'invert_neighbors')
    def __init__(self, idx, x, y):
        self.idx = idx
        self.position = complex(x,y)
        self.neighbors = {}
        self.edges = []
        self.invert_neighbors = {}

    def __repr__(self,):
        return "Node <%i>" % self.idx

    def add_edge(self, edge):
        self.edges.append(edge)
        self.neighbors[edge.stop] = edge
        edge.stop.invert_neighbors[self] = edge

    def __getitem__(self, stop):
        return self.neighbors[stop]

class Graph(object):

    def __init__(self,):
        self.nodes = []
        self.edge_cost = {}

    def reset(self,):
        for edge in self.edges():
            edge.reset()


    def add_edge(self, edge):
        edge.start.add_edge(edge)

    def edges(self,):
        for node in self.nodes:
            for edge in node.edges:
                yield edge

    def add_node(self, node_id, x, y):
        assert node_id >= 0
        for n in range(len(self.nodes), node_id+1):
            self.nodes.append(None)    
        self.nodes[node_id] = Node(node_id, x, y)

    def __getitem__(self, n):
        return self.nodes[n]


def parse():
    g = Graph()
    with open('paris_54000.txt') as f:
        N, M, T, C, S = map(int, f.readline().split())
        for node_id in xrange(N):
            x, y = map(float, f.readline().split())
            g.add_node(node_id, a*x, a*y)
        for j in range(M):
            start, stop, bothways, cost, distance = map(int, f.readline().split())
            start = g[start]
            stop = g[stop]
            fwd_edge = Edge(idx=j, start=start, stop=stop, cost=cost, distance=distance,)
            g.add_edge(fwd_edge)
            if bothways==2:
                bwd_edge = Edge(idx=j, start=stop, stop=start, cost=cost, distance=distance,)
                bwd_edge.reverse = fwd_edge
                fwd_edge.reverse = bwd_edge
                g.add_edge(bwd_edge)
    return g



if __name__ == '__main__':
    parse()
