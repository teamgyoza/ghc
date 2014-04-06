class Edge:

    __slots__ = ( 'distance', 'cost', 'start', 'stop')
    def __init__(self, start, stop, distance, cost):
        self.distance = distance
        self.cost = cost
        self.start = start
        self.stop = stop

class Node:

    __slots__ = ('id', 'edges', 'x', 'y')
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.edges = {}

class Graph:

    def __init__(self,):
        self.nodes = []

    def add_edge(self, edge):
        self.nodes[edge.start].edges[self.end] = edge

    def add_node(self, node_id, x, y):
        assert node_id >= 0
        for n in range(len(self.nodes), node_id+1):
            self.nodes.append(None)    
        self.nodes[node_id] = Node(id, x, y)

    def __item__(self, n):
        return self.nodes[n]



def parse():
    g = Graph()
    with open('paris_54000.txt') as f:
        N, M, T, C, S = map(int, f.readline().split())
        for node_id in xrange(N):
            g.add_node(node_id)
            x, y = map(float, f.readline().split())
            g.add_node(node_id, x, y)
        for j in range(M):
            start, stop, bothways, cost, distance = map(int, f.readline().split())
            g.add_edge(Edge(id=j, start=start, stop=stop, cost=cost, distance=distance))
            if bothways == 2:
                g.add_edge(Edge(id=j, start=start, stop=stop, cost=cost, distance=distance))
    return g


def submit(car_paths, filepath='output.txt'):
    total_cars = 8
    source_id = 4516
    f = open(filepath, "w")
    lines = []
    lines.append(total_cars)
    for car_id in range(total_cars):
        if car_id < len(car_paths):
            lines.append(len(car_paths[car_id]))
            for intersection_id in car_paths[car_id]:
                lines.append(intersection_id)
        else:
            lines.append(1)
            lines.append(source_id)
    f.write('\n'.join(map(str, lines)))

if __name__ == '__main__':
    parse()
