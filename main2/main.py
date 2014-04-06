from graph import parse
import sys
import random
from math import sqrt

DEPTH=4
START=4516
TIME=54000

# obtained by linear regression to have
# euclidian dist match distances
a = 114180.046772
b = 111803.550422

def project(graph):
    """ exchanged lat,long coordinates
    for a simple projection on a map that
    gives ok estimates of distances.
    """
    for n in graph.nodes:
        n.x = a*n.x
        n.y = b*n.y


# class Score(object):

#     def __init__(self,):
#         self.distance = distance
#         self.cost = cost
#         self.length = length
        

#     def


class Car(object):

    __slots__ = ('id', 'position', 'timeleft', 'path', 'distance')

    def __init__(self, id, position, timeleft=TIME):
        self.id = id
        self.position = position
        self.timeleft = timeleft
        self.path = [position]
        self.distance = 0

    def move(self, stop):
        self.path.append(stop)
        self.position = stop

    def follow(self, edge):
        self.move(edge.stop)
        self.path.append(edge.stop.idx)
        self.position = edge.stop
        self.distance += edge.distance
        self.timeleft -= edge.cost

        

def choose_best_edge(g, car):
    """ returns the best score, with the matching edge. """
    candidates = list(browse_candidates(g, car.position, car.timeleft, set(), DEPTH))
    if not candidates:
        return None
    #return max(candidates)[1]
    best_score = max(candidates)[0]
    # print best_score
    #return max(candidates)[1]
    best_candidates = [ candidate
            for (score, candidate) in candidates
            if score == best_score]
    return random.choice(best_candidates)
    


def browse_candidates(g, node, timeleft, visited, depth):
    if depth == 0:
        for edge in node.edges:
            if edge.cost <= timeleft:
                score = 0.
                if edge.idx not in visited:
                    score = edge.efficiency
                yield (score, edge)
    else:
        for edge in node.edges:
            if edge.cost <= timeleft:
                dest = edge.stop
                score = 0.
                if dest.idx not in visited:
                    score += edge.efficiency
                candidates = list(browse_candidates(g, dest, timeleft - edge.cost, visited.union([dest.idx]), depth-1))
                if candidates:
                    yield (score + max(candidates)[0], edge)
                elif score > 0.:
                    yield (score, edge)

def traverse(g, cars):
    while any(car.timeleft > 0 for car in cars):
        for car in cars:
            if car.timeleft > 0:
                edge = choose_best_edge(g, car)
                if edge:
                    car.follow(edge)
                    edge.visit()
                else:
                    car.timeleft = 0
    print "overall distance", sum(car.distance for car in cars)
    return cars

def run(filepath='output.txt'):
    g = parse()
    project(g)
    cars = [
        Car(id=car_id, position=g[START])
        for car_id in range(8)
    ]
    traverse(g, cars)


if __name__ == '__main__':
    run(*sys.argv[1:])
