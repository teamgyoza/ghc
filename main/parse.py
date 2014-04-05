from collections import namedtuple

import networkx as nx


def parse():
    G = nx.DiGraph()

    with open('paris_54000.txt') as f:
        N, M, T, C, S = map(int, f.readline().split())
        print N, M, T, C, S

        for i in range(N):
            x, y = map(float, f.readline().split())
            G.add_node(i, i=i, x=x, y=y)

        for j in range(M):
            a, b, d, c, l = map(int, f.readline().split())
            G.add_edge(a, b, j=j, start=a, stop=b, cost=c, distance=l)
            if d == 2:
                G.add_edge(b, a, j=j, start=b, stop=a, cost=c, distance=l)

    return dict(G=G, N=N, M=M, T=T, C=C, S=S)

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
