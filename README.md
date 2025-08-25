# Case (a): good init near natural clusters
python kmeans_1d.py db/t_kmeans_1d.csv 3 --init 1,11,28

# Case (b): poor init farther from natural clusters
python kmeans_1d.py db/t_kmeans_1d.csv 3 --init 1,2,3
