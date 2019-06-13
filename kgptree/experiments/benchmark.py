import rdflib
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import sys
sys.path.append('..')
from tree_builder import KGPTree
from datastructures import *
import time

import pickle

def train_model(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, output):
	print(end='Loading data... ', flush=True)
	g = rdflib.Graph()
	if format is not None:
		g.parse(rdf_file, format=format)
	else:
		g.parse(rdf_file)
	print('OK')

	test_data = pd.read_csv(train_file, sep='\t')
	train_data = pd.read_csv(test_file, sep='\t')

	train_entities = [rdflib.URIRef(x) for x in train_data[entity_col]]
	train_labels = train_data[label_col]

	test_entities = [rdflib.URIRef(x) for x in test_data[entity_col]]
	test_labels = test_data[label_col]

	kg = KnowledgeGraph.rdflib_to_kg(g, label_predicates=label_predicates)

	start = time.time()
	#clf = GridSearchCV(KGPTree(kg, path_max_depth=6, neighborhood_depth=8), scoring='accuracy',
	#				   {'max_tree_depth': [5, 7, 10, 13, None], 'min_samples_leaf': [1, 3, 5, 10]}, 
	#				   cv=StratifiedKFold(n_splits=5))
	clf = KGPTree(kg, path_max_depth=6, neighborhood_depth=8, max_tree_depth=None, min_samples_leaf=1)
	clf.fit(train_entities, train_labels)

	results = {}
	#results['params'] = clf.best_params_
	results['time'] = time.time() - start

	preds = clf.predict(test_entities)

	results['accuracy'] = accuracy_score(test_labels, preds)
	results['confusion_matrix'] = confusion_matrix(test_labels, preds)

	pickle.dump(results, open(output, 'wb+'))	

	print('Test accuracy = {} || Confusion Matrix:'.format(accuracy_score(test_labels, preds)))
	print(confusion_matrix(test_labels, preds))



##################### AIFB #####################################
rdf_file = '../data/AIFB/aifb.n3'
format = 'n3'
train_file = '../data/AIFB/AIFB_test.tsv'
test_file = '../data/AIFB/AIFB_train.tsv'
entity_col = 'person'
label_col = 'label_affiliation'
label_predicates = [
    rdflib.URIRef('http://swrc.ontoware.org/ontology#affiliation'),
    rdflib.URIRef('http://swrc.ontoware.org/ontology#employs'),
    rdflib.URIRef('http://swrc.ontoware.org/ontology#carriedOutBy')
]
output = 'output/aifb_no_tuning.p'
train_model(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, output)

###################### BGS #####################################
rdf_file = '../data/BGS/BGS.nt'
format = 'nt'
train_file = '../data/BGS/BGS_test.tsv'
test_file = '../data/BGS/BGS_train.tsv'
entity_col = 'rock'
label_col = 'label_lithogenesis'
label_predicates = [
    rdflib.term.URIRef('http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis'),
    rdflib.term.URIRef('http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesisDescription'),
    rdflib.term.URIRef('http://data.bgs.ac.uk/ref/Lexicon/hasTheme')
]
output = 'output/bgs_no_tuning.p'
train_model(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, output)


##################### MUTAG ####################################
rdf_file = '../data/MUTAG/mutag.owl'
format = None
train_file = '../data/MUTAG/MUTAG_test.tsv'
test_file = '../data/MUTAG/MUTAG_train.tsv'
entity_col = 'bond'
label_col = 'label_mutagenic'
label_predicates = [
    rdflib.term.URIRef('http://dl-learner.org/carcinogenesis#isMutagenic')
]
output = 'output/mutag_no_tuning.p'
train_model(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, output)


###################### AM ######################################
rdf_file = '../data/AM/rdf_am-data.ttl'
format = 'turtle'
train_file = '../data/AM/AM_test.tsv'
test_file = '../data/AM/AM_train.tsv'
entity_col = 'proxy'
label_col = 'label_cateogory'
label_predicates = [
    rdflib.term.URIRef('http://purl.org/collections/nl/am/objectCategory'),
    rdflib.term.URIRef('http://purl.org/collections/nl/am/material')
]
output = 'output/am_no_tuning.p'
train_model(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, output)
