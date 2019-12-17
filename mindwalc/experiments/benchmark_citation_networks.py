import rdflib
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append('..')
from tree_builder import KGPTree, KGPForest, KPGTransformer
from datastructures import *
import time

from itertools import product
from collections import defaultdict

import pickle

import warnings; warnings.filterwarnings('ignore')

def train_model(rdf_file, train_entities, test_entities, dev_entities, labels, label_predicates, output):
    # Load in our graph using rdflib
    print(end='Loading data... ', flush=True)
    g = rdflib.Graph()
    g.parse(rdf_file, format='turtle')
    print('OK')

    print(end='Converting to KG... ', flush=True)
    kg = Graph.rdflib_to_graph(g, label_predicates=label_predicates)
    print('OK')

    labels = pd.read_csv(labels, sep='\t', header=None, index_col=0)
    train_ids = [x.strip() for x in open(train_entities, 'r').readlines()]
    test_ids = [x.strip() for x in open(test_entities, 'r').readlines()]
    val_ids = [x.strip() for x in open(dev_entities, 'r').readlines()]

    train_labels = [str(labels.loc[int(i)][1]) for i in train_ids]
    test_labels = [str(labels.loc[int(i)][1]) for i in test_ids]
    val_labels = [str(labels.loc[int(i)][1]) for i in val_ids]

    train_entities = [rdflib.URIRef('http://paper_'+x) for x in train_ids]
    test_entities = [rdflib.URIRef('http://paper_'+x) for x in test_ids]
    val_entities = [rdflib.URIRef('http://paper_'+x) for x in val_ids]

    results = {}
    results['ground_truth'] = test_labels

    # Tune Transform + RF
    params = {
        'path_max_depth': [4, 6, 8, 10],
        'n_features': [1000, 3000, 5000, 7500, 10000, 20000]
    }

    best_params, best_score = None, 0
    combinations = list(itertools.product(*list(params.values())))
    for combination in combinations:
        param_combination = dict(zip(params.keys(), combination))
        transf = KPGTransformer(n_jobs=-1, **param_combination)

        transf.fit(kg, train_entities, train_labels)

        train_features = transf.transform(kg, train_entities)
        val_features = transf.transform(kg, val_entities)

        useful_features = np.sum(train_features, axis=0) > 1

        train_features = train_features[:, useful_features]
        val_features = val_features[:, useful_features]

        clf = GridSearchCV(RandomForestClassifier(random_state=42, max_features=None), 
                   {'n_estimators': [10, 100, 250], 'max_depth': [5, 10, None]})
        clf.fit(train_features, train_labels)
        preds = clf.predict(val_features)

        acc = accuracy_score(val_labels, preds)
        print('RF+Transform', param_combination, acc)

        if acc > best_score:
            best_score = acc
            best_params = param_combination

    transf = KPGTransformer(n_jobs=-1, **best_params)
    transf.fit(kg, train_entities, train_labels)

    train_features = transf.transform(kg, train_entities)
    test_features = transf.transform(kg, test_entities)

    clf = GridSearchCV(RandomForestClassifier(random_state=42, max_features=None), 
               {'n_estimators': [10, 100, 250], 'max_depth': [5, 10, None]})
    clf.fit(train_features, train_labels)
    preds = clf.predict(test_features)

    results['transform_rf_preds'] = preds

    # Tune Transform + LR

    params = {
        'path_max_depth': [4, 6, 8, 10],
        'n_features': [1000, 3000, 5000, 7500, 10000, 20000]
    }

    best_params, best_score = None, 0
    combinations = list(itertools.product(*list(params.values())))
    for combination in combinations:
        param_combination = dict(zip(params.keys(), combination))
        transf = KPGTransformer(n_jobs=-1, **param_combination)

        transf.fit(kg, train_entities, train_labels)

        train_features = transf.transform(kg, train_entities)
        val_features = transf.transform(kg, val_entities)

        useful_features = np.sum(train_features, axis=0) > 1

        train_features = train_features[:, useful_features]
        val_features = val_features[:, useful_features]

        clf = GridSearchCV(LogisticRegression(random_state=42), 
                {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]})
        clf.fit(train_features, train_labels)
        preds = clf.predict(val_features)

        acc = accuracy_score(val_labels, preds)
        print('LR+Transform', param_combination, acc)

        if acc > best_score:
            best_score = acc
            best_params = param_combination

    transf = KPGTransformer(n_jobs=-1, **best_params)
    transf.fit(kg, train_entities, train_labels)

    train_features = transf.transform(kg, train_entities)
    test_features = transf.transform(kg, test_entities)

    clf = GridSearchCV(LogisticRegression(random_state=42), 
                {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]})
    clf.fit(train_features, train_labels)
    preds = clf.predict(test_features)

    results['transform_lr_preds'] = preds

    # For forest, tune the vertex_sample, max_tree_depth and n_estimators
    params = {
        'max_tree_depth': [5, None],
        'vertex_sample': [0.5, 0.9]
    }

    best_params, best_score = None, 0
    combinations = list(itertools.product(*list(params.values())))
    for combination in combinations:
        param_combination = dict(zip(params.keys(), combination))
        accuracies = {}

        clf = KGPForest(path_max_depth=8, n_jobs=-1, n_estimators=50, 
                        **param_combination)
        clf.fit(kg, train_entities, train_labels)

        for n_estimators in [10, 25, 50]:
            clf_dummy = KGPForest(path_max_depth=8)
            clf_dummy.estimators_ = clf.estimators_[:n_estimators]
            preds = clf_dummy.predict(kg, val_entities)

            accuracies[n_estimators] = accuracy_score(val_labels, preds)

        for n_estimators in [10, 25, 50]: 
            if accuracies[n_estimators] > best_score:
                best_score = accuracies[n_estimators]
                param_combination['n_estimators'] = n_estimators
                best_params = param_combination

    print('Tuned params = {}'.format(best_params))

    clf = KGPForest(path_max_depth=8, n_jobs=-1, **best_params)
    clf.fit(kg, train_entities, train_labels)
    preds = clf.predict(kg, test_entities)

    print('[Forest] Test accuracy = {} || Confusion Matrix:'.format(accuracy_score(test_labels, preds)))
    print(confusion_matrix(test_labels, preds))

    results['forest_preds'] = preds

    # Tune the max_tree_depth
    best_depth, best_score = None, 0
    for depth in [3, 5, 10, None]:
        clf = KGPTree(path_max_depth=8, max_tree_depth=depth, n_jobs=-1)
        clf.fit(kg, train_entities, train_labels)
        preds = clf.predict(kg, val_entities)
        acc = accuracy_score(val_labels, preds)
        if acc > best_score:
            best_score = acc
            best_depth = depth

    print('Tuned depth = {}'.format(best_depth))

    clf = KGPTree(path_max_depth=8, max_tree_depth=best_depth, min_samples_leaf=1, n_jobs=-1)
    clf.fit(kg, train_entities, train_labels)
    preds = clf.predict(kg, test_entities)
    
    print('[Tree] Test accuracy = {} || Confusion Matrix:'.format(accuracy_score(test_labels, preds)))
    print(confusion_matrix(test_labels, preds))

    results['tree_preds'] = preds

    output_file = '{}_{}.p'.format(output[:-2], int(time.time()))
    pickle.dump(results, open(output_file, 'wb+'))




for _ in range(10):

    ##################### CITESEER ####################################
    rdf_file = 'data/CITESEER/citeseer.ttl'
    label_file = 'data/CITESEER/label.txt'
    train_file = 'data/CITESEER/train.txt'
    test_file = 'data/CITESEER/test.txt'
    val_file = 'data/CITESEER/dev.txt'
    label_predicates = [
        rdflib.URIRef('http://hasLabel')
    ]
    output = 'output/citeseer.p'
    train_model(rdf_file, train_file, test_file, val_file, label_file, label_predicates, output)

    ##################### CORA ####################################
    rdf_file = 'data/CORA/cora.ttl'
    label_file = 'data/CORA/label.txt'
    train_file = 'data/CORA/train.txt'
    test_file = 'data/CORA/test.txt'
    val_file = 'data/CORA/dev.txt'
    label_predicates = [
        rdflib.URIRef('http://hasLabel')
    ]
    output = 'output/cora.p'
    train_model(rdf_file, train_file, test_file, val_file, label_file, label_predicates, output)

    ##################### PUBMED ####################################
    rdf_file = 'data/PUBMED/pubmed.ttl'
    label_file = 'data/PUBMED/label.txt'
    train_file = 'data/PUBMED/train.txt'
    test_file = 'data/PUBMED/test.txt'
    val_file = 'data/PUBMED/dev.txt'
    label_predicates = [
        rdflib.URIRef('http://hasLabel')
    ]
    output = 'output/pubmed.p'
    train_model(rdf_file, train_file, test_file, val_file, label_file, label_predicates, output)