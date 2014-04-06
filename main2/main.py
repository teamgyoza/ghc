from graph import parse
import sys
import random
from math import sqrt, pi
from heapq import heappush, heappop

DEPTH=4
ORIGIN=4516
TIME=54000


class Car(object):

    __slots__ = ('id', 'position', 'timeleft', 'path', 'edges', 'distance', 'angle', 'running')

    def __init__(self, id, position, timeleft=TIME):
        self.id = id
        self.position = position
        self.timeleft = timeleft
        self.path = [position]
        self.edges = []
        self.distance = 0
        self.angle = 0.
        self.running = True

    def follow_path(self, path):
        for edge in path:
            self.follow(edge)

    def follow(self, edge):
        assert self.position == edge.start
        self.edges.append(edge)
        self.path.append(edge.stop)
        self.position = edge.stop
        self.distance += edge.distance
        self.timeleft -= edge.cost
        if self.timeleft == 0:
            self.running = False
        self.angle = edge.angle
        edge.visit()

    def go_to(self, g, dest):
        path = path_to(g, self.position, dest)
        self.follow_path(path)
        assert self.position == dest

    def revisit(self, g):
        # rerun the path, and mark the graph
        # edges as visited
        self.distance = 0
        for edge in self.edges:
            self.distance += edge.distance
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

def djikstra(g, start, end):
    visited = set()
    frontier = [ (0, start) ]
    distances = {}
    while frontier:
        (cost, n) = heappop(frontier)
        if n in visited:
            continue
        if n == end:
            return (distances, cost)
        distances[n] = cost
        for edge in n.edges:
            if edge.stop not in visited:
                candidate = (cost + edge.cost, edge.stop)
                heappush(frontier, candidate)
        visited.add(n)
    return None

def path_to(g, start, end):
    sol = djikstra(g, start, end)
    if sol is None:
        return None
    (distances, cost) = sol
    return compute_path(distances, cost, start, end)



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
    while any(car.timeleft > 0 and car.running for car in cars):
        for car in cars:
            if car.running and car.timeleft > 0:
                path = choose_best_path(g, car)
                if path is None:
                    car.running = False
                else:
                    car.follow_path(path)
    return cars

def mean(l):
    return sum(l) / float(len(l))

import time

def search():
    m = 0
    runs = []
    times = []
    g = parse()
    while True:
        start = time.time()
        g.reset()
        start_point_ids = [ random.randint(0, len(g.nodes) - 1) for i in range(7) ] + [ ORIGIN ]
        (score, paths) = run(g,start_point_ids)
        end = time.time()
        runs.append(score)
        times.append(end-start)
        #if len(runs) % 10 == 0:
        print mean(runs), mean(times), len(runs)
        if score > m:
            m = score
            print start_point_ids, m
            submit(paths, "test.txt")

def overall_score(cars):
    return sum(car.distance for car in cars)

def run(g, start_point_ids):
    origin = g[ORIGIN]
    cars = [
        Car(id=car_id, position=origin)
        for car_id in range(8)
    ]
    # route cars to their starting points
    for (car, start_point_id) in zip(cars, start_point_ids):
        start_point = g[start_point_id]
        car.go_to(g, start_point)
        assert car.position == start_point
    traverse(g, cars)
    score = overall_score(cars)
    if score > 1915312:
        postprocess(g, cars)
        score = overall_score(cars)
    return score, [car.path for car in cars]

#--------------------------
# postprocessing

def find_shortcuts(g, cars,):
    for car_id in range(8):
        g.reset()
        new_cars = [
            Car(id=car_id, position=g[ORIGIN])
            for car_id in range(8)
        ]
        for i in range(8):
            if i != car_id:
                new_cars[i].follow_path(cars[i].edges)
        cur = cars[car_id]
        a = 0
        new_car_path = []
        while a < len(cur.edges):
            b = a
            while b<len(cur.edges):
                if cur.edges[b].distance > 0:
                    break
                b += 1
            assert all(edge.distance == 0 for edge in cur.edges[a:b])
            if b-a > 1:
                start = cur.edges[a].start
                stop = cur.edges[b-1].stop
                assert sum(edge.distance for edge in cur.edges[a:b]) == 0
                section = path_to(g, start, stop)
                if section:
                    assert start == section[0].start
                    assert stop == section[-1].stop
                new_car_path += section
                a=b
            else:
                new_car_path.append(cur.edges[a])
                a += 1
            assert sum(edge.distance for edge in new_car_path) >= sum(edge.distance for edge in cur.edges[0:a])
        assert cur.edges[0].start == new_car_path[0].start
        assert cur.edges[-1].stop == new_car_path[-1].stop
        new_cars[car_id].follow_path(new_car_path)
        #print overall_score(new_cars)
        cars[:] = new_cars[:]

def postprocess(g, cars,):
    print "------"
    print "postprocess"
    print "Best score", overall_score(cars)
    print "margin", sum(car.timeleft for car in cars)
    find_shortcuts(g, cars)
    print "margin", sum(car.timeleft for car in cars)
    print "After shorcuts score", overall_score(cars)
    traverse(g, cars)
    print "final", overall_score(cars)

    
            
            


#if __name__ == '__main__':
#   g = parse()
#    print run(g, [4645, 10137, 6872, 9290, 4470, 2101, 1847, 4516])[0]


if __name__ == '__main__':
    print search()