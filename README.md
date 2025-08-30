/db
db\processed
db\raw

model
models\classification
models\cluster
models\pattern


python start.py apriori db/raw/tx.json 0.4 0.6


PATTERNmodels\pattern\apriori.py
python models/pattern/apriori.py db/raw/tx.json 0.4 0.6
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

python Classification/trees/id3.py db/id3.json label
python Classification/trees/c45.py db/id3.json label
python Classification/trees/c50.py db/id3.json label
// Uses Bayes’ Rule p(c|x)
python Classification/naive_bayes.py db/t_employee.json department=systems status=junior age=26..30


# single query
python classification_models/nb_cli.py predict --model nb_employee.json --query department=systems status=junior age=26..30

# batch (make sure db/queries.json exists)
python classification_models/nb_cli.py predict --model nb_employee.json --batch db/queries.json
python -m classification_models.naivebayes.nb_cli train db/t_employee.json --label salary
