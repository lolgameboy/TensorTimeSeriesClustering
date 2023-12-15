import math
import random
import numpy as np
from numpy import mean


def k_means(k, data, means=[]):
    # Add sufficient means
    if len(means) < k:
        means.extend(random.sample(data, k))

    # Create points as tuples (index of cluster, value)
    points = []
    for d in data:
        points.append([None, np.array(d)])

    # Create clusters as tuples (mean, list of points)
    clusters = []
    for m in means:
        clusters.append([m, []])

    # Main loop
    changed = True
    while changed:
        changed = False
        # Assign points to closest mean
        for p in points:
            min_dist = math.inf
            closest_mean = None
            for m in range(0, len(clusters)):
                dist = np.linalg.norm(p[1]-clusters[m][0])
                if dist < min_dist:
                    min_dist = dist
                    closest_mean = m
            if p[0] != closest_mean:
                changed = True
                if p[0] is not None:
                    clusters[p[0]][1].remove(p)
                clusters[closest_mean][1].append(p)
                p[0] = closest_mean

        # Calculate new means
        for c in clusters:
            v_list = []
            for p in c[1]:
                v_list.append(p[1])
            if len(v_list) == 0:
                break
            c[0] = mean(v_list, axis=0)
        print(clusters)
    return clusters

d = [[1,2],[4,5],[7,8],[8,8],[1,1],[4,8],[9,7],[2,3],[1,4],[1,4],[6,4],[9,0],[1,2]]
r = k_means(5, d)
print(r)
