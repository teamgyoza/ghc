import random

from parse import parse, submit
import sys
from networkx import shortest_path
from collections import defaultdict

from collections import defaultdict
from multiprocessing import cpu_count, Pool
from random import choice

from parse import parse, submit


def traverse(G, T, C, start_points):
    intersections = [ [] for i in range(8) ]
    current = start_points
    distance = [0] * 8
    timeleft = [T] * 8

    while any(timeleft[i] > 0 for i in range(C)):
        print timeleft
        for i in range(C):
            if timeleft[i] > 0:
                edge = choose(G, current[i], timeleft[i])

                if not edge:
                    timeleft[i] = 0
                    break

                start, stop = edge['start'], edge['stop']

                current[i] = stop
                distance[i] += edge['distance']
                timeleft[i] -= edge['cost']
                edge['distance'] = 0

                try:
                    G[stop][start]['distance'] = 0
                except KeyError:
                    pass

                intersections[i].append(current[i])

    distance = sum(distance.values())
    return distance, intersections.values()


def choose(G, current, timeleft):
    edges = G[current].values()
    candidates = [(score(G, edge, timeleft), edge) for edge in edges if edge['cost'] <= timeleft]
    if not candidates:
        return None
    best, _ = max(candidates)
    equals = [e for s, e in candidates if s == best]
    return choice(equals)

def score(G, edge, timeleft):
    return weight(edge) + score_estimate(G, edge['stop'], { edge['j'] },
            depth=10)

def weight(edge):
    return edge['distance'] / float(edge['cost'])

def score_estimate(G, current, visited, depth):
    gain = 0.
    edges = G[current].values()
    if depth == 0:
        return gain
    else:
        gain += max(
            (weight(edge) if (edge['j'] not in visited) else 0)
            + score_estimate(G, edge['stop'],  visited.union({edge['j']}), depth-1)
            for edge in edges
        )
    return gain
  
        

def choose_depth(G,current,timeleft):
    edges = G[current].values()
    

#def search_score(G, edge, timeleft):



# if __name__ == '__main__':
#     paris = parse()
#     traversals = [traverse(paris['G'], paris['S'], paris['T']) for i in range(8)]
#     distance = sum([d for _, d in traversals])
#     print distance
#     paths = [i for i, _ in traversals]
#     submit(paths)





def search_path(g, start, dest):
    path = shortest_path(g, start, dest, weight='cost')
    cost = sum(
        g[a][b]['cost']
        for (a,b) in zip(path, path[1:])
    )
    distance = sum(
        g[a][b]['distance']
        for (a,b) in zip(path, path[1:])
    )
    for (a,b) in zip(path, path[1:]):
        g[a][b]['distance'] = 0
        try:
            g[b][a]['distance'] = 0
        except KeyError:
            pass
    return cost, distance, path#[:-1]

def run(filepath='output.txt'):
    paris = parse()
    g = paris['G']
    traversals = []
    S = paris['S']
    T = paris['T']
    
    start_points = [
        random.randint(0,11340)
        for i in range(8)
    ]

    print start_points
    start_points = [ 2745, 4416, 1149, 1904, 3848, 3735, 2307, 5064]
    # start_points = [3575, 6214, 10490, 895, 7600, 23, 5832, 6562]
    
    start_points = [10071, 7925, 677, 4661, 9109, 5654, 7543, 4269]
    start_points = [S] * 8 
    #[157, 9761, 8551, 1896, 9901, 484, 9677, 7853]
    #start_points = [8723, 1949, 2587, 4561, 1741, 1700, 9182, 2024]
    #print start_points
    distance = 0
    Ts = [ T ] * 8
    for car in range(8):
        #weight_graph(g, car)
        path = []
        start_point = start_points[car]
        (cost, dist, path) = search_path(g, S, start_point)
        Ts[car] -= cost
        distance += dist
        #distance += dist
        traversals.append(path)
    #for car in range(8):
    dist_next, traversals_next = traverse(g, Ts[car], 8, start_points)
    distance += dist_next
    for car in range(8):
        #distance += dist
        #path +=  traversal
        traversals[car] += traversals_next[car]
    print distance
    #
    #print sum( map(distance, traversals))
    #distance = sum([d for _, d in traversals])
    #print distance
    #paths = [i for i, _ in traversals]
    submit(traversals, filepath)
    return distance

if __name__ == '__main__':
    import sys
    run(sys.argv[1])
