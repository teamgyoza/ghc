from graph import parse
import sys
import random
from math import sqrt, pi
from heapq import heappush, heappop
import numpy as np

DEPTH=4
ORIGIN=4516
TIME=54000


# class Score(object):

#     def __init__(self,):
#         self.distance = distance
#         self.cost = cost
#         self.length = length
        

#     def


class Car(object):

    __slots__ = ('id', 'position', 'timeleft', 'path', 'distance', 'angle')

    def __init__(self, id, position, timeleft=TIME):
        self.id = id
        self.position = position
        self.timeleft = timeleft
        self.path = [position]
        self.distance = 0
        self.angle = 0.

    def move(self, stop):
        self.path.append(stop)
        self.position = stop

    def follow_path(self, path):
        for edge in path:
            self.follow(edge)

    def follow(self, edge):
        assert self.position == edge.start
        self.path.append(edge.stop)
        self.position = edge.stop
        self.distance += edge.distance
        self.timeleft -= edge.cost
        self.angle = edge.angle
        edge.visit()

        

def choose_best_path(g, car):
    """ returns the best score, with the matching edge. """
    #return browse_candidates(g, car.position, car.timeleft, set())
    return closest(g, car.position, car.angle, car.timeleft)



def diff_angle(a1, a2):
    da = a1 - a2
    if da < -pi:
        da += 2.*pi
    if da > pi:
        da -= 2.*pi
    #print da
    assert -pi <= da <= pi
    return da


def submit(car_paths, filepath='output.txt'):
    total_cars = 8
    source_id = 4516
    f = open(filepath, "w")
    lines = []
    lines.append(total_cars)
    for car_id in range(total_cars):
        if car_id < len(car_paths):
            lines.append(len(car_paths[car_id]))
            for intersection in car_paths[car_id]:
                lines.append(intersection.idx)
        else:
            lines.append(1)
            lines.append(source_id)
    f.write('\n'.join(map(str, lines)))


def compute_path(distances, cost, start, stop):
    if start == stop:
        return []
    else:
        for (node,edge) in stop.invert_neighbors.items():
            #print distances
            if node in distances:
                if edge.cost + distances[node] == cost:
                    return compute_path(distances, cost - edge.cost, start, node) + [ edge ]
        raise "could not find path"


def path_to(g, start, end):
    visited = set()
    frontier = [ (0, start) ]
    distances = {}
    while frontier:
        (cost, n) = heappop(frontier)
        if n in visited:
            continue
        if n == end:
            return compute_path(distances, cost, start, end)
        distances[n] = cost
        for edge in n.edges:
            if edge.stop not in visited:
                candidate = (cost + edge.cost, edge.stop)
                heappush(frontier, candidate)
        visited.add(n)
    return None


def priority(in_angle):
    def aux(edge):
        #return -edge.distance / float(edge.cost)
        #return diff_angle(edge.angle, in_angle)
        return (edge.reverse is not None, -edge.distance / float(edge.cost))
    return aux

def closest(g, start, incoming_angle, timeleft):
    visited = set()
    frontier = [ (0, incoming_angle, start) ]
    distances = {}
    while frontier:
        (cost, in_angle, n) = heappop(frontier)
        if n in visited:
            continue
        distances[n] = cost
        if cost > timeleft:
            # cannot reach a non visited edge on time
            return None
        edges = sorted(n.edges, key=priority(in_angle))
        for edge in edges:
            if cost + edge.cost <= timeleft:
                # we can take this edge
                if edge.distance > 0:
                    return compute_path(distances, cost, start, n) + [ edge ]
                else:
                    if edge.stop not in visited:
                        candidate = (cost + edge.cost, edge.angle, edge.stop)
                        # print candidate
                        heappush(frontier, candidate)
        visited.add(n)
    return None


def traverse(g, cars):
    while any(car.timeleft > 0 for car in cars):
        for car in cars:
            if car.timeleft > 0:
                path = choose_best_path(g, car)
                if path is None:
                    car.timeleft = 0
                else:
                    car.follow_path(path)
    return cars

def search():
    m = 0
    runs = []
    while True:
        g = parse()
        start_point_ids = [ random.randint(0, len(g.nodes)) for i in range(7) ] + [ ORIGIN ]
        (score, paths) = run(g,start_point_ids)
        runs.append(score)
        arr = np.array(runs)
        print arr.mean(), arr.std(), np.median(arr), len(runs)
        if score > m:
            m = score
            print start_point_ids, m
            submit(paths, "output3.txt")


def run(g, start_point_ids):
    g = parse()
    origin = g[ORIGIN]
    cars = [
        Car(id=car_id, position=origin)
        for car_id in range(8)
    ]
    # route cars to their starting points
    for (car, start_point_id) in zip(cars, start_point_ids):
        start_point = g[start_point_id]
        path = path_to(g, origin, start_point)
        car.follow_path(path)
        assert car.position == start_point
    traverse(g, cars[0:7])
    traverse(g, cars[7:8])
    """
    for i in range(7):
        traverse(g, cars[i:i+1])
    """
    score = sum(car.distance for car in cars)
    return score, [car.path for car in cars]


#if __name__ == '__main__':
#   g = parse()
#    print run(g, [4645, 10137, 6872, 9290, 4470, 2101, 1847, 4516])[0]


if __name__ == '__main__':
    print search()