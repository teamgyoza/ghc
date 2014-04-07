from graph import parse
import sys
import random
from math import sqrt, pi
from heapq import heappush, heappop

DEPTH=7
ORIGIN=4516
TIME=54000

# 1898984.09031 1.30129733585 18502 1925016

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
    source_id = 4516
    f = open(filepath, "w")
    lines = []
    lines.append(8)
    for car_path in car_paths:
        lines.append(len(car_path))
        for intersection in car_path:
            lines.append(intersection.idx)
    f.write('\n'.join(map(str, lines)))



def compute_path(distances, cost, start, stop):
    if start == stop:
        return []
    else:
        for (node,edge) in stop.invert_neighbors.items():
            if node in distances:
                if edge.cost + distances[node] == cost:
                    return compute_path(distances, cost - edge.cost, start, node) + [ edge ]
        raise "could not find path"

def djikstra(g, start, end):
    visited = set()
    frontier = [ ( (0,0), start) ]
    distances = {}
    while frontier:
        ((cost, count), n) = heappop(frontier)
        if n in visited:
            continue
        if n == end:
            return (distances, cost)
        distances[n] = cost
        for edge in n.edges:
            if edge.stop not in visited:
                candidate = ((cost + edge.cost, count+edge.visits), edge.stop)
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
        # diff_angle(in_angle, edge.angle) 
        return (edge.reverse is not None, -edge.distance / float(edge.cost), )
        # return (edge.reverse is not None,)
    return aux

def closest(g, start, incoming_angle, timeleft):
    visited = set()
    frontier = [ (0, 0, incoming_angle, start) ]
    distances = {}
    while frontier:
        (cost, count, in_angle, n) = heappop(frontier)
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
                        candidate = (cost + edge.cost, count + edge.visits, edge.angle, edge.stop)
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
    # 8551, 4598, 6919, 3296, 4277, 9557, 1874, 4516]#
    while True:
        start = time.time()
        g.reset()
        start_point_ids = [ random.randint(0, len(g.nodes) - 1) for i in range(7) ] + [ ORIGIN ]
        # random.start_point_ids = [ 8551, 4598, 6919, 3296, 4277, 9557, 1874, 4516]
        (score, paths) = run(g,start_point_ids)
        end = time.time()
        runs.append(score)
        times.append(end-start)
        print mean(runs), mean(times), len(runs), m
        if score > m:
            m = score
            print start_point_ids, m
            submit(paths, "testc.txt")

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
    #traverse(g, cars[7:8])
    score = overall_score(cars)
    #if score > 1915312:
    from itertools import count
    for r in count(1):
        print "Round % i", r
        postprocess(g, cars)
        new_score = overall_score(cars)
        if new_score <= score:
            break
        else:
            score = new_score
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
        cars[:] = new_cars[:]


def browse_deviation(start, burnt, path_budgets, max_cost, depth=DEPTH):
    if depth == 0:
        return
    if max_cost == 0:
        return
    for edge in start.edges:
        if edge not in burnt:
            budget = max_cost - edge.cost
            if edge.stop in path_budgets:
                (distance, subpath) = path_budgets[edge.stop](budget) # distance for the remaining path given a budget
                yield (edge.distance + distance, [edge] + subpath)
            else:
                possible_deviations = list(browse_deviation(edge.stop, burnt.union({edge, edge.reverse}), path_budgets, budget, depth-1))
                if possible_deviations:
                    (distance, suffix) = max(possible_deviations)
                    yield (distance + edge.distance, [edge] + suffix)

"""
def simple_path_budget(path):
    def aux(budget):
        d = 0
        for e in path:
            if e.cost <= budget:
                d += e.distance
                budget -= e.cost
            else:
                break
        return d
    return aux
"""

def simple_path_budget(path):
    def aux(budget):
        subpath = []
        for e in path:
            if e.cost <= budget:
                subpath.append(e)
                budget -= e.cost
            else:
                break
        return (sum(e.distance for e in subpath), subpath)
    return aux

def compute_path_budgets(path):
    c = {}
    for (i,e) in enumerate(path):
        c[e.start] = simple_path_budget(path[i:])
    return c

def deviate_path(g, path):
    burnt = set(e for e in path).union(e.reverse for e in path)
    budget = TIME
    prefix_distance = 0
    prefix = []
    path_budgets = compute_path_budgets(path)
    for i,e in enumerate(path):
        for (suffix_distance, suffix) in browse_deviation(e.start, burnt, path_budgets, budget, depth=DEPTH):
            yield (prefix_distance + suffix_distance, prefix + suffix)
        # we can't go backward in the path
        if e.start in path_budgets:
            del path_budgets[e.start]
        budget -= e.cost
        prefix.append(e)
        prefix_distance += e.distance

def valid_path(path):
    for (a,b) in zip(path, path[1:]):
        assert b.start == a.stop

def extend_with_path(path, former_path):
    intersection = path[-1].stop
    intersection_last_idx = -1
    for (i,e) in enumerate(former_path):
        if e.start == intersection:
            intersection_last_idx = i
    assert intersection_last_idx != -1
    path += former_path[intersection_last_idx:]



def deviate_postprocess(g, cars):
    for car_id in range(8):
        print "optimizing car %i" % car_id
        g.reset()
        new_cars = [
            Car(id=i, position=g[ORIGIN])
            for i in range(8)
        ]
        for i in range(8):
            if i != car_id:
                new_cars[i].follow_path(cars[i].edges)
        target = sum(e.distance for e in cars[car_id].edges)
        deviations = list(deviate_path(g, cars[car_id].edges))
        if deviations:
            (d, path) = max(deviations)
            assert d == sum(e.distance for e in path)
            valid_path(path)
            new_cars[car_id].follow_path(path)
            cars[:] = new_cars[:]
        else:
            print "no deviations"




def postprocess(g, cars,):
    print "------"
    print "POSTPROCESS"
    print "Best score", overall_score(cars)
    print "Margin", sum(car.timeleft for car in cars)
    find_shortcuts(g, cars)
    traverse(g, cars)
    print "Ater shortcuts", overall_score(cars)
    deviate_postprocess(g, cars)
    print "final", overall_score(cars)



def load_solution(filepath="best.txt"):
    with open(filepath, 'r') as f:
        lines = [ int(line.strip())  for line in f.readlines() ][1:]
        line_it = iter(lines)
        car_paths = []
        for i in range(8):
            count = line_it.next()
            path = []
            for c in range(count):
                path.append(line_it.next())
            car_paths.append(path)
        return car_paths

def main():
    solution = load_solution()

def score(g, car_paths):
    g.reset()
    for car_path in car_paths:
        cars = [
            Car(id=car_id, position=g[ORIGIN])
                for car_id in range(8)
        ]
    for (car, car_path) in zip(cars, car_paths):
        car.follow_path(car_path)
    return sum(car.distance for car in cars)

if __name__ == '__main__':
    solution = load_solution("best2.txt")
    g = parse()
    cars = [
        Car(id=car_id, position=g[ORIGIN])
            for car_id in range(8)
    ]
    for (intersections, car) in zip(solution, cars):
        for intersection in intersections[1:]:
            car.follow(car.position[g.nodes[intersection]])
    score = overall_score(cars)
    from itertools import count
    for r in count(1):
        print "Round % i" % r
        postprocess(g, cars)
        new_score = overall_score(cars)
        submit([car.path for car in cars], "postproces-2s%i.txt" % r)
        print new_score
        if new_score <= score:
            break
        else:
            score = new_score
    print score(g, [car.edges for car in cars ])
    submit([car.path for car in cars], "postprocess.txt")

    


#if __name__ == '__main__':
#    print search()