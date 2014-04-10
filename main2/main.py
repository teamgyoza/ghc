from graph import parse
import sys
import random
from math import sqrt, pi
from heapq import heappush, heappop

DEPTH=9
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
        edge.visit(self,)

    def go_to(self, g, dest):
        path = path_to(self.position, dest)
        self.follow_path(path)
        assert self.position == dest

    def revisit(self, g):
        # rerun the path, and mark the graph
        # edges as visited
        self.distance = 0
        for edge in self.edges:
            self.distance += edge.distance
            edge.visit(self,)


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
    print "writing to ", filepath
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

def djikstra(start, end):
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

def path_to(start, end):
    """
    Return the shortest path from start to end.
    """
    sol = djikstra(start, end)
    if sol is None:
        return None
    (distances, cost) = sol
    return compute_path(distances, cost, start, end)



def priority(in_angle):
    def aux(edge):
        return (edge.reverse is not None, -edge.distance / float(edge.cost), )
    return aux

def closest(start, incoming_angle, timeleft):
    """
    Find the closest non-visited street
    with a priority described in traverse
    """
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
    """
    Traverse drives the car using a simple heuristic.
    Basically drives road that have not been driven yet,
    with a priority to taking oneway street and 
    to turning left.

    When not available, just go to the closest 
    non-visited street.
    """
    while any(car.timeleft > 0 and car.running for car in cars):
        for car in cars:
            if car.running and car.timeleft > 0:
                path = closest(car.position, car.angle, car.timeleft)
                if path is None:
                    car.running = False
                else:
                    car.follow_path(path)
    return cars

def mean(l):
    return sum(l) / float(len(l))

import time

def search(output_filepath):
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
            submit(paths, output_filepath)

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
    cars = traverse(g, cars)
    score = overall_score(cars)
    #if score > 1915312:
    """
    from itertools import count
    for r in count(1):
        print "Round % i", r
        postprocess(g, cars, 10)
        new_score = overall_score(cars)
        if new_score <= score:
            break
        else:
            score = new_score
    """
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
                section = path_to(start, stop)
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


def browse_deviation(start, burnt, path_budgets, budget, depth=DEPTH):
    if depth == 0:
        return
    for edge in start.edges:
        if edge not in burnt and edge.cost <= budget:
            new_budget = budget - edge.cost
            if edge.stop in path_budgets:
                # distance for the remaining path given a budget
                (distance, margin) = path_budgets[edge.stop](new_budget) 
                yield (edge.distance + distance, margin, [edge])
            else:
                possible_deviations = list(browse_deviation(edge.stop, burnt.union({edge, edge.reverse}), path_budgets, new_budget, depth-1))
                if possible_deviations:
                    (distance, margin, suffix) = max(possible_deviations)
                    yield (distance + edge.distance, margin, [edge] + suffix)



def simple_path_budget(path):
    def aux(budget):
        """
        Given a budget how much
        of a path can you run?       
        """
        d = 0
        travelled = 0
        pending_travelled = 0
        remaining = budget
        for e in path:
            if e.cost <= remaining:
                remaining -= e.cost
                pending_travelled += e.cost
                if e.distance > 0:
                    d += e.distance
                    travelled += pending_travelled
                    pending_travelled = 0
            else:
                break
        return (d, budget - travelled)
    return aux

def compute_margin(path, timeleft=TIME):
    return simple_path_budget(path)(timeleft)[1]

def compute_path_budgets(path):
    """
    Given a path, returns a dictionary
    which for each node, return the following function
        
        (budget)-> distance that can be reached
                   by following this path up to the given budget
    
    """
    c = {}
    for (i,e) in enumerate(path):
        c[e.start] = simple_path_budget(path[i:])
    return c

def deviate_path(g, path, depth):
    """
    Generator on deviation opportunities\
    Opportunities have the shape (distance, path)
    """
    (original_distance, original_margin) = simple_path_budget(path)(TIME)
    burnt = set(e for e in path).union(e.reverse for e in path)
    budget = TIME
    prefix_distance = 0
    prefix = []
    path_budgets = compute_path_budgets(path)
    for i,e in enumerate(path):
        for (suffix_distance, margin, suffix) in browse_deviation(e.start, burnt, path_budgets, budget, depth=depth):
            distance = prefix_distance + suffix_distance
            if (distance, margin) > (original_distance, original_margin):
                yield (distance, margin, prefix + suffix)
        # we can't go backward in the path
        if e.start in path_budgets:
            del path_budgets[e.start]
        budget -= e.cost
        prefix.append(e)
        prefix_distance += e.distance

def valid_path(path):
    for (a,b) in zip(path, path[1:]):
        assert b.start == a.stop

def extend_with_path(path, former_path, budget):
    """
    Given 2 interecting paths, extends in place
    the first one with the second one until
    the budget is depleted.
    """
    intersection = path[-1].stop
    intersection_last_idx = -1
    for edge in path:
        budget -= edge.cost
    for (i,e) in enumerate(former_path):
        if e.start == intersection:
            intersection_last_idx = i
    assert intersection_last_idx != -1
    segment = []
    for edge in former_path[intersection_last_idx:]:
        if edge.cost <= budget:
            segment.append(edge)
            budget -= edge.cost
            if edge.distance > 0:
                path+=segment
                segment = []
            
        else:
            break


#def get_gradient(cars, delta_budget):
#    pass

def deviate_postprocess(g, cars, depth):
    """
    Given a solution (cars),
    modifies it in place in a better solution by 
    looping on cars and searching for
    opportunities to locally deviate from
    the original trajectory.
    """
    for car_id in range(8):
        print "optimizing car %i" % car_id,
        former_score = overall_score(cars)
        g.reset()
        new_cars = [
            Car(id=i, position=g[ORIGIN])
            for i in range(8)
        ]
        for i in range(8):
            if i != car_id:
                new_cars[i].follow_path(cars[i].edges)
        target = sum(e.distance for e in cars[car_id].edges)
        deviations = list(deviate_path(g, cars[car_id].edges, depth))
        new_score = overall_score(cars)
        if deviations:
            (d, margin, path) = max(deviations)
            extend_with_path(path, cars[car_id].edges, TIME)
            assert d == sum(e.distance for e in path)
            new_cars[car_id].follow_path(path)
            cars[:] = new_cars[:]
            new_score = overall_score(cars)
            print new_score - former_score, margin
        else:
            print "no deviations"


def postprocess(g, cars, depth):
    print "------"
    print "POSTPROCESS"
    print "Best score", overall_score(cars)
    margin = sum(car.timeleft for car in cars)
    print "MARGIN", margin
    assert margin >= 0
    find_shortcuts(g, cars)
    traverse(g, cars)
    print "Ater shortcuts", overall_score(cars)
    deviate_postprocess(g, cars, depth)
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

def get_intersections(path):
    return [ start_node.neighbors[stop_node]
             for start_node, stop_node in zip(path[:-1], path[1:])]

def score_solution(solution):
    g=parse()
    car_paths = [get_intersections([g[node_id] for node_id in node_ids])
                 for node_ids in solution]
    solution_score = score(g, car_paths)
    print "score:", solution_score
    return solution_score

def score(g, car_paths):
    g.reset()
    cars = [
        Car(id=car_id, position=g[ORIGIN])
            for car_id in range(8)
    ]
    for (car, car_path) in zip(cars, car_paths):
        car.follow_path(car_path)
    return overall_score(cars)

def optimize_postprocessing(depth, input_filepath, output_filepath):
    depth = int(depth)
    solution = load_solution(input_filepath)
    g=parse()
    print "total length", sum(e.distance for e in g.edges())
    cars = [
        Car(id=car_id, position=g[ORIGIN])
        for car_id in range(8)
    ]
    for (intersections, car) in zip(solution, cars):
        for intersection in intersections[1:]:
            car.follow(car.position[g.nodes[intersection]])
    #traverse(g, cars)
    from itertools import count
    for r in count(1):
        random.shuffle(cars)
        print "Round % i" % r
        former_score = overall_score(cars)
        former_timeleft = sum(car.timeleft for car in cars)
        postprocess(g, cars, depth)
        new_score = overall_score(cars)
        new_timeleft = sum(car.timeleft for car in cars)
        total_margin = sum(compute_margin(car.edges) for car in cars)
        print "total margin", total_margin
        submit([car.path for car in cars], output_filepath + str(new_score))
        print new_score
        if (new_score, new_timeleft) <= (former_score, former_timeleft):
            break
    print score

def show_help():
    print """
    Usage:
        main.py postprocess <depth> <input_filepath> <output_filepath>
        main.py search <output_filepath>
        main.py score <solution_filepath>
    """

if __name__ == '__main__':
    if len(sys.argv) < 2:
        show_help()
    else:
        command_name, args = sys.argv[1], sys.argv[2:]
        {
            'postprocess': optimize_postprocessing,
            'search': search,
            'score': lambda f: score_solution(load_solution(f))
        }.get(command_name, show_help)(*args)
