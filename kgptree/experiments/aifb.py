import rdflib
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

import sys
sys.path.append('..')
from tree_builder import KGPTree
from datastructures import *

print(end='Loading data... ', flush=True)
g = rdflib.Graph()
g.parse('../data/AIFB/aifb.n3', format='n3')
print('OK')

label_predicates = [
    rdflib.URIRef('http://swrc.ontoware.org/ontology#affiliation'),
    rdflib.URIRef('http://swrc.ontoware.org/ontology#employs'),
    rdflib.URIRef('http://swrc.ontoware.org/ontology#carriedOutBy')
]

test_data = pd.read_csv('../data/AIFB/AIFB_test.tsv', sep='\t')
train_data = pd.read_csv('../data/AIFB/AIFB_train.tsv', sep='\t')

train_people = [rdflib.URIRef(x) for x in train_data['person']]
train_labels = train_data['label_affiliation']

test_people = [rdflib.URIRef(x) for x in test_data['person']]
test_labels = test_data['label_affiliation']

kg = KnowledgeGraph.rdflib_to_kg(g, label_predicates=label_predicates)

clf = GridSearchCV(KGPTree(kg, path_max_depth=4, neighborhood_depth=8, min_samples_leaf=1), {'max_tree_depth': [3, 5, 10, None]}, cv=3)
clf.fit(train_people, train_labels)

preds = clf.predict(test_people)
print('Test accuracy = {} || Confusion Matrix:'.format(accuracy_score(test_labels, preds)))
print(confusion_matrix(test_labels, preds))