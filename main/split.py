from cmath import phase
from math import pi
from collections import Counter
import numpy as np

x_0=48.877846700000006
y_0=2.3269475

def angle(x, y):
    rad_x = x - x_0
    rad_y = y - y_0
    theta = phase(complex(rad_x, rad_y))
    normalized_theta = (theta + pi) / (2. * pi)
    return normalized_theta

def district(angles):
    def aux(x,y):
        a = angle(x,y)
        for (i,bucket) in enumerate(angles):
            if bucket >= a:
                return max(0, i-1)
        return 7
    return aux

def get_weight_angle(angles, car):
    angles = np.array(angles)
    A = len(angles)
    car_median = int(A * car / 8.0)
    angles = np.array(angles)
    def aux(x,y):
        a = angle(x,y)
        idx = np.searchsorted(angles, a)
        dist = abs(car_median - idx)
        if dist > A / 2:
            dist -= A / 2
        dist /= float(A)
        if dist < 0.5 / 8.:
            return 1.
        else:
            return 0.
    return aux


    

def add_district(g):
    angles = []
    for i in g.nodes_iter():
        node = g.node[i]
        x = node['x']
        y = node['y']
        angles.append(angle(x,y))
    angles.sort()
    A = len(angles)
    limits = []
    for i in range(8):
        limits.append(angles[ A * i/ 8])
    get_district = district(limits)
    for i in g.nodes_iter():
        node = g.node[i]
        node['district'] = get_district(x,y)


def weight_graph(g, car):
    angles = []
    for i in g.nodes_iter():
        node = g.node[i]
        x = node['x']
        y = node['y']
        angles.append(angle(x,y))
    angles.sort()
    weight_angle = get_weight_angle(angles, car)
    for (_,_,edge) in g.edges_iter(data=True):  
        end = g.node[edge["stop"]]
        x = end["x"]
        y = end["y"]
        weight = weight_angle(x,y)
        edge["weight"] = weight
    return g




if __name__=="__main__":
    from parse import parse
    g = parse()["G"]
    weight_graph(g, 1)
