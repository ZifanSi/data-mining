# default minsup=0.5, minconf=0.7
python apriori.py db/tx.json

# custom thresholds
python apriori.py db/tx.json 0.4 0.6


# default minsup=0.5
python fp_growth.py db/tx.json

# custom minsup (e.g., 0.6)
python fp_growth.py db/tx.json 0.6



# Case (a): good init near natural clusters
python kmeans_1d.py db/t_kmeans_1d.csv 3 --init 1,11,28

# Case (b): poor init farther from natural clusters
python kmeans_1d.py db/t_kmeans_1d.csv 3 --init 1,2,3
