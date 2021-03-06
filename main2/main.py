from graph import *
import sys
import random
from itertools import count
from collections import Counter, defaultdict
from math import sqrt, pi
from heapq import heappush, heappop

DEPTH=9
ORIGIN=4516
TIME=54000

class Car(object):

    __slots__ = ('id', 'position', 'timeleft', 'intersections', 'edges', 'distance', 'angle', 'running')

    def __init__(self, id, position, timeleft=TIME):
        self.id = id
        self.position = position
        self.timeleft = timeleft
        self.intersections = [position]
        self.edges = []
        self.distance = 0
        self.angle = 0.
        self.running = True

    def follow_path(self, path, budget=None):
        if budget is None:
            budget = TIME*1000
        for edge in path:
            if edge.cost <= budget:
                budget -= edge.cost
                self.follow(edge)
            else:
                break
        return (self.distance, budget)

    def follow(self, edge):
        assert self.position == edge.start
        self.edges.append(edge)
        self.intersections.append(edge.stop)
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

    def marginal_score(self,):
        """Distance covered only by this car.

        This is also the score increase
        we gain if we remove and add the car back.
        """
        score = 0
        visited = set()
        for edge in self.edges:
            if edge not in visited:
                visited.add(edge)
                visited.add(edge.reverse)
                if len(edge.cars) == 1:
                    score += edge.original_distance
        return score


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
        intersections = path_to_intersections(car_path)
        lines.append(len(intersections))
        for intersection in intersections:
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

def djikstra_set(start, A):
    visited = set()
    frontier = [ ( (0,0), start) ]
    distances = {}
    while frontier:
        ((cost, count), n) = heappop(frontier)
        if n in visited:
            continue
        if n in A:
            return (cost, n)
        distances[n] = cost
        for edge in n.edges:
            if edge.stop not in visited:
                candidate = ((cost + edge.cost, count+edge.visits), edge.stop)
                heappush(frontier, candidate)
        visited.add(n)
    return None

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
        return (edge.difficulty, -edge.distance / float(edge.cost), )
    return aux

def closest(start, incoming_angle, timeleft):
    """
    Find the closest non-visited street
    with a priority described in traverse
    """
    visited = set()
    frontier = [ (0, 0, 0, incoming_angle, start) ]
    distances = {}
    while frontier:
        (cost, difficulty, count, in_angle, n) = heappop(frontier)
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
                        difficulty = max(e2.difficulty for e2 in edge.stop.edges)
                        candidate = (cost + edge.cost, difficulty, count + edge.visits, edge.angle, edge.stop)
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
    print "compute_difficulty"
    g.compute_difficulty()
    print "done"
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
    return score, [car.edges for car in cars]

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

def max_or_none(g):
    M = None
    for el in g:
        if el > M:
            M = el
    return M


def browse_deviation(start, burnt, path_suffix_map, budget, depth=DEPTH):
    if depth == 0:
        return
    for edge in start.edges:
        if edge.cost <= budget:
            gain = 0
            relax7_edge = 0
            oneway_edge = 0
            if edge.idx not in burnt:
                gain = edge.distance
                if edge.cars == {7}:
                    relax7_edge += 1
                if edge.reverse is None:
                    oneway_edge += 1
            new_budget = budget - edge.cost
            new_burnt = burnt.union({edge.idx})
            if edge.stop in path_suffix_map:
                # distance for the remaining path given a budget
                suffix = path_suffix_map[edge.stop]
                (oneway, distance, relax7, margin) = compute_distance_and_margin(suffix, new_budget, new_burnt) #path_suffix_map[edge.stop](new_budget, new_burnt) 
                yield (oneway + oneway_edge, gain + distance, relax7_edge+ relax7, margin, [edge])
            best_deviation = max_or_none(browse_deviation(edge.stop, new_burnt, path_suffix_map, new_budget, depth-1))
            if best_deviation:
                (oneway, distance, relax7, margin, suffix) = best_deviation
                yield (oneway + oneway_edge, gain + distance,  relax7_edge + relax7, margin, [edge] + suffix)
            elif budget < 100:
                if gain == 0:
                    yield (0, 0, 0, budget, [])
                else:
                    yield (0, 0, 0, new_budget, [])


def compute_distance_and_margin(path, budget, burnt):
    distance = 0
    travelled = 0
    remaining = budget
    pending_travelled = 0
    oneway = 0
    visited = set(burnt)
    relax7 = 0
    for edge in path:
        if edge.cost <= remaining:
            remaining -= edge.cost
            pending_travelled += edge.cost
            if edge.idx not in visited:
                visited.add(edge.idx)
                if edge.distance > 0:
                    if edge.reverse is None:
                        oneway += 1
                    distance += edge.distance
                    travelled += pending_travelled
                    pending_travelled = 0
                if edge.cars == {7}:
                    relax7 += 1
        else:
            break
    return (oneway, distance, relax7, budget - travelled)


def compute_margin(path, timeleft=TIME):
    return compute_distance_and_margin(path, timeleft, set())[2]

def deviate_path(g, path, depth):
    """
    Generator on deviation opportunities\
    Opportunities have the shape (distance, path)
    """
    budget = TIME
    prefix_distance = 0
    prefix = []
    prefix_visited = set()
    prefix_relax7 = 0
    prefix_oneway = 0
    path_suffix_map = {
        e.start: path[i:]
        for (i,e) in enumerate(path)
    }
    for i,edge in enumerate(path):
        for (suffix_one_way, suffix_distance, suffix_relax7, margin, suffix) in browse_deviation(edge.start, prefix_visited, path_suffix_map, budget, depth=depth):
            distance = prefix_distance + suffix_distance
            relax7 = prefix_relax7 + suffix_relax7
            oneway = prefix_oneway + suffix_one_way
            yield (oneway, distance, relax7, margin, prefix + suffix)
        budget -= edge.cost
        prefix.append(edge)
        if edge.start in path_suffix_map:
            del path_suffix_map[edge.start]
        if edge.idx not in prefix_visited:
            prefix_visited.add(edge.idx)
            prefix_distance += edge.distance
            if edge.reverse is None:
                prefix_oneway += 1
            if edge.cars == {7}:
                prefix_relax7 += 1

def assert_valid_path(path):
    for (a,b) in zip(path, path[1:]):
        assert b.start == a.stop

def deviate_postprocess(g, cars, depth):
    """
    Given a solution (cars),
    modifies it in place in a better solution by 
    looping on cars and searching for
    opportunities to locally deviate from
    the original trajectory.
    """
    print "\n"*3
    for car_id in range(8):
        print "-------------"
        print " optimizing car %i" % cars[car_id].id
        start = time.time()
        g.reset()
        new_cars = [
            Car(id=i, position=g[ORIGIN])
            for i in range(8)
        ]
        for i in range(8):
            if i != car_id:
                new_cars[i].follow_path(cars[i].edges)
        original_path = cars[car_id].edges
        (former_one_way, former_distance, former_relax7, former_margin) = compute_distance_and_margin(original_path, TIME, {})
        best_deviation = max_or_none(deviate_path(g, original_path, depth))
        if best_deviation:
            (oneway, d, r7, m, prefix) = best_deviation
            
            middle_point = prefix[-1].stop
            suffix = {
                e.start: original_path[i:]
                for (i,e) in enumerate(original_path)
            }.get(middle_point, [])
            new_path = prefix + suffix
            (new_one_way, new_distance, new_relax7, new_margin) = compute_distance_and_margin(new_path, TIME, {})
            print "onway",oneway, new_one_way
            new_cars[car_id].follow_path(new_path, TIME)
            assert d == new_distance
            if (new_one_way, new_distance, new_relax7, new_margin) > (former_one_way, former_distance, former_relax7, former_margin):
                print "   one_way  :", new_one_way - former_one_way
                print "  distance gain :", new_distance - former_distance
                print "    margin gain :", new_margin - former_margin
                cars = new_cars
            else:
                print "no deviations"
        else:
            print "no deviations"
        end = time.time()
        print "took ", (end-start)
    return cars


def postprocess(g, cars, depth):
    print "------"
    print "POSTPROCESS"
    print "Best score", overall_score(cars)
    margin = sum(car.timeleft for car in cars)
    print "MARGIN", margin
    assert margin >= 0
    find_shortcuts(g, cars)
    traverse(g, cars)
    print "AFTER SHORTCUTS", overall_score(cars)
    return  deviate_postprocess(g, cars, depth)

def load_solution(g, filepath="best.txt"):
    paths = []
    with open(filepath, 'r') as f:
        lines = ( int(line.strip())  for line in f.readlines() )
        lines.next()
        intersection_lists = []
        for i in range(8):
            count = lines.next()
            intersection_list = []
            for c in range(count):
                intersection = g.nodes[lines.next()]
                intersection_list.append(intersection)
            path = intersections_to_path(intersection_list)
            paths.append(path)
    return paths

def intersections_to_path(intersections):
    return [ start_node.neighbors[stop_node]
             for start_node, stop_node in zip(intersections, intersections[1:])]

def path_to_intersections(path):
    intersections = []
    for edge in path:
        intersections.append(edge.start)
    intersections.append(edge.stop)
    return intersections


def search_opportunities(g, paths):
    print "\n"*5
    unvisited = [
        edge
        for edge in g.edges()
        if edge.distance > 0 and (edge.reverse is None or edge.start < edge.stop)
    ]
    print "nb streets unvisited", len(unvisited)
    oneway = {True:0, False:0}
    for edge in unvisited:
        oneway[edge.reverse is None] += 1
    print "Oneway?"
    for (k,v) in oneway.items():
        print k,v
    print "----------------"
    path_intersections_list = []
    for (i,path) in enumerate(paths):
        path_intersections = { e.start for e in path }.union({path[-1].stop})
        path_intersections_list.append(path_intersections)
    for edge in unvisited:
        print "--------"
        print "Edge cost",edge.cost
        for (i,path_intersections) in enumerate(path_intersections_list):
            if edge.start in path_intersections:
                (d, node) = djikstra_set(edge.stop, {edge.start})
                print "Distance to path %i " % i, d + edge.cost
                #print i, djikstra_set(edge.start, path_intersections)


def score_solution(g, paths):
    g.reset()
    cars = [
        Car(id=car_id, position=g[ORIGIN])
            for car_id in range(8)
    ]
    for (car, path) in zip(cars, paths):
        car.follow_path(path)
    score = overall_score(cars)
    print "score    :  ", score
    print "--------------------"
    print "\n"
    for (car_id, car) in enumerate(cars):
        print " marginal ", car_id, ":", car.marginal_score()
    print "\n"
    for (car_id, car) in enumerate(cars):
        print " margin ", car_id, ":", car.timeleft
    print " total margin ", ":", sum(car.timeleft for car in cars)
    print "\n" * 3
    inverse_graph = defaultdict(list)

    oneway_count = {False: 0, True:0}
    for edge in g.edges():
        inverse_graph[edge.stop].append(edge)
        if edge.reverse is None or edge.start<edge.stop:
            oneway_count[edge.reverse is None] += 1
    print "One way overall"
    for k,v in oneway_count.items():
        print k,v

    counter_outgoing = Counter()
    counter_incoming = Counter()
    ahead_only = 0
    for node in g.nodes:
        counter_outgoing[len(node.edges)] += 1
        counter_incoming[len(inverse_graph[node])] += 1
        if len(node.edges) == len(inverse_graph[node]) == 1:
            assert node.edges[0] != inverse_graph[node][0].reverse # no dead ends
            ahead_only += 1
            #print node.edges[0], inverse_graph[node][0]
    print "\n"*3
    print "Number of possible exit per intersection"
    for k,v in counter_outgoing.items():
        print k, v
    print "\n"*3
    print "Number of possible entry per intersection"
    for k,v in counter_incoming.items():
        print k, v
    print "\n"*3
    print "Number of dead ends"
    print 0
    print "\n"*3
    print "Number of ahead only"
    print ahead_only
    print "Max length", sum(e.distance for e in g.edges() if e.reverse < e)
    print "Number of nodes", len(g.nodes)

    search_opportunities(g, paths)

def optimize_postprocessing(depth, input_filepath, output_filepath):
    depth = int(depth)
    g=parse()
    paths = load_solution(g, input_filepath)

    cars = [
        Car(id=car_id, position=g[ORIGIN])
        for car_id in range(8)
    ]
    for (path, car) in zip(paths, cars):
        car.follow_path(path)

    for r in count(1):
        print "Round % i" % r
        random.shuffle(cars)
        #sort(car.margin for car in)
        #cars.sort(key=lambda car:-car.timeleft)
        for car in cars:
            print car.id, ":", car.timeleft
        former_score = overall_score(cars)
        former_timeleft = sum(car.timeleft for car in cars)
        cars = postprocess(g, cars, depth)
        new_score = overall_score(cars)
        new_timeleft = sum(car.timeleft for car in cars)
        total_margin = sum(compute_margin(car.edges) for car in cars)
        print new_score
        if (new_score, new_timeleft) <= (former_score, former_timeleft):
            print "Increasing depth", depth
            depth += 1
        else:
            print "New Score ", new_score
            submit([car.edges for car in cars], output_filepath + str(new_score))



def command_score(f):
    g = parse()
    solution = load_solution(g, f)
    score_solution(g, solution)

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
            'score': command_score
        }.get(command_name, show_help)(*args)
