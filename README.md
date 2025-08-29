PATTERN
python pattern/apriori.py db/tx.json 0.4 0.6
python pattern/fp_growth.py db/tx.json 0.6
python pattern/hmine.py db/tx.json 0.5
python pattern/rule_measures.py db/rules.json

CLUSTER
python cluster/kmeans_1d.py db/k.json 3

UTILS
python utils/descriptive_stats.py db/age_freq.json

AGNES (2 clusters, average-link):
python cluster/hier.py agnes db/points.json --k 2 --link average


DIANA (3 clusters):
python cluster/hier.py diana db/points.json --k 3

BIRCH (radius=2.0):
python cluster/hier.py birch db/points.json --radius 2.0
python cluster\bicluster.py

python Classification/id3.py db/id3.json label
python Classification/c45.py db/id3.json label
python Classification/c50.py db/id3.json label
python Classification/naive_bayes.py db/t_employee.json department=systems status=junior age=26..30
