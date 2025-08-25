#!/usr/bin/env python3
# Minimal 1D K-Means (pure Python, no external libs)
# Usage:
#   python kmeans_1d.py db/t_kmeans_1d.csv 3 --init 1,11,28
#   python kmeans_1d.py db/t_kmeans_1d.csv 3 --maxiter 100

import sys

def parse_args(argv):
    if len(argv) < 3:
        print("Usage: python kmeans_1d.py <data_csv> <k> [--init a,b,c] [--maxiter N]")
        sys.exit(1)
    path = argv[1]
    k = int(argv[2])
    init = None
    maxiter = 100
    i = 3
    while i < len(argv):
        if argv[i] == "--init" and i+1 < len(argv):
            init = [float(x) for x in argv[i+1].split(",") if x.strip()!=""]
            i += 2
        elif argv[i] == "--maxiter" and i+1 < len(argv):
            maxiter = int(argv[i+1]); i += 2
        else:
            i += 1
    return path, k, init, maxiter

def read_1d_csv(path):
    nums = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                nums.extend([float(x) for x in line.split(",") if x.strip()!=""])
            else:
                nums.append(float(line))
    return nums

def assign(points, centers):
    clusters = [[] for _ in centers]
    for x in points:
        idx = min(range(len(centers)), key=lambda j: abs(x - centers[j]))
        clusters[idx].append(x)
    return clusters

def recompute(clusters, old_centers):
    new_centers = []
    for j, cluster in enumerate(clusters):
        if cluster:
            new_centers.append(sum(cluster)/len(cluster))
        else:
            new_centers.append(old_centers[j])
    return new_centers

def kmeans_1d(points, k, init=None, maxiter=100):
    points = list(points)
    points.sort()
    if init is None:
        step = max(1, len(points)//k)
        centers = [points[min(i*step, len(points)-1)] for i in range(k)]
    else:
        centers = list(init)
        if len(centers) != k:
            raise ValueError("len(--init) must equal k")
    history = []
    for it in range(1, maxiter+1):
        clusters = assign(points, centers)
        history.append((it, centers[:], [c[:] for c in clusters]))
        new_centers = recompute(clusters, centers)
        if all(abs(a - b) < 1e-9 for a, b in zip(centers, new_centers)):
            centers = new_centers
            history.append(("converged", centers[:], [c[:] for c in clusters]))
            break
        centers = new_centers
    return centers, clusters, history

def main():
    path, k, init, maxiter = parse_args(sys.argv)
    data = read_1d_csv(path)
    centers, clusters, history = kmeans_1d(data, k, init=init, maxiter=maxiter)
    for record in history:
        it, centers, clusters = record
        if it == "converged":
            print(f"\n[Converged] centers = {centers}")
        else:
            print(f"\nIteration {it}:")
            for j, c in enumerate(centers, 1):
                pts = clusters[j-1]
                print(f"  Cluster {j}: points={pts}  center={c:.2f}")
    print("\nFinal centers:", centers)
    for j, pts in enumerate(clusters, 1):
        print(f"Cluster {j}: {pts}  -> center={centers[j-1]:.2f}")

if __name__ == "__main__":
    main()
