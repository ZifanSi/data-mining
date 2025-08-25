PATTERN
python pattern/apriori.py db/tx.json 0.4 0.6
python pattern/fp_growth.py db/tx.json 0.6
python pattern/hmine.py db/tx.json 0.5
python pattern/rule_measures.py db/rules.json

CLUSTER
python cluster/kmeans_1d.py db/k.json 3

UTILS
python utils/descriptive_stats.py db/age_freq.json
