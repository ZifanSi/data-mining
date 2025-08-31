/db
db\processed
db\raw

model
models\classification
models\cluster
models\pattern

// Mining
python start.py hmine   db/raw/tx.json 0.4
python start.py fp      db/raw/tx.json 0.4
python start.py apriori db/raw/tx.json 0.4 0.6

// Distillation
python process/tx_norm.py db/raw/tx.json
python process/tx_norm.py db/raw/tx.json


CLUSTER
python models/cluster/kmeans_1d.py db/raw/k.json 3 

UTILS
python utils/descriptive_stats.py db/age_freq.json

AGNES (2 clusters, average-link):
python cluster/hier.py agnes db/points.json --k 2 --link average


DIANA (3 clusters):
python cluster/hier.py diana db/points.json --k 3

BIRCH (radius=2.0):
python cluster/hier.py birch db/points.json --radius 2.0
python cluster\bicluster.py

python Classification/trees/id3.py db/id3.json label
python Classification/trees/c45.py db/id3.json label
python Classification/trees/c50.py db/id3.json label
// Uses Bayes’ Rule p(c|x)
python Classification/naive_bayes.py db/t_employee.json department=systems status=junior age=26..30



