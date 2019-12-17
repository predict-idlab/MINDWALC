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

def train_model(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, output):
    # Load in our graph using rdflib
    print(end='Loading data... ', flush=True)
    g = rdflib.Graph()
    if format is not None:
        g.parse(rdf_file, format=format)
    else:
        g.parse(rdf_file)
    print('OK')

    # Create some lists of train and test entities & labels
    test_data = pd.read_csv(train_file, sep='\t')
    train_data = pd.read_csv(test_file, sep='\t')

    train_entities = [rdflib.URIRef(x) for x in train_data[entity_col]]
    train_labels = train_data[label_col]

    test_entities = [rdflib.URIRef(x) for x in test_data[entity_col]]
    test_labels = test_data[label_col]

    # Convert the rdflib graph to our graph
    kg = Graph.rdflib_to_graph(g, label_predicates=label_predicates)

    results = {}
    results['ground_truth'] = test_labels

    transf = KPGTransformer(path_max_depth=8, n_features=10000, n_jobs=-1)
    start = time.time()
    transf.fit(kg, train_entities, train_labels)
    transf_fit_time = time.time() - start

    train_features = transf.transform(kg, train_entities)
    test_features = transf.transform(kg, test_entities)

    useful_features = np.sum(train_features, axis=0) > 1

    train_features = train_features[:, useful_features]
    test_features = test_features[:, useful_features]

    clf = GridSearchCV(RandomForestClassifier(random_state=42, max_features=None), 
               {'n_estimators': [10, 100, 250], 'max_depth': [5, 10, None]})
    clf.fit(train_features, train_labels)
    transf_rf_preds = clf.predict(test_features)

    print('[Transform + Random Forest] Test accuracy = {} || Confusion Matrix:'.format(accuracy_score(test_labels, transf_rf_preds)))
    print(confusion_matrix(test_labels, transf_rf_preds))

    clf = GridSearchCV(LogisticRegression(random_state=42, penalty='l1'), 
                {'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]})
    clf.fit(train_features, train_labels)
    transf_lr_preds = clf.predict(test_features)

    print('[Transform + Logistic Regression] Test accuracy = {} || Confusion Matrix:'.format(accuracy_score(test_labels, transf_lr_preds)))
    print(confusion_matrix(test_labels, transf_lr_preds))

    results['transform_fit_time'] = transf_fit_time
    results['transform_lr_preds'] = transf_lr_preds
    results['transform_rf_preds'] = transf_rf_preds

    N_SPLITS = 5

    # For forest, tune the vertex_sample, max_tree_depth and n_estimators
    params = {
        'max_tree_depth': [5, None],
        'vertex_sample': [0.5, 0.9]
    }

    best_params, best_score = None, (0, 0)
    combinations = list(itertools.product(*list(params.values())))
    for combination in combinations:
        param_combination = dict(zip(params.keys(), combination))
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        accuracies = defaultdict(list)
        for train_ix, test_ix in cv.split(train_entities, train_labels):
            cv_train_entities = [train_entities[ix] for ix in train_ix]
            cv_train_labels = [train_labels[ix] for ix in train_ix]
            cv_test_entities = [train_entities[ix] for ix in test_ix]
            cv_test_labels = [train_labels[ix] for ix in test_ix]

            clf = KGPForest(path_max_depth=8, n_jobs=-1, n_estimators=50, 
                            **param_combination)
            clf.fit(kg, cv_train_entities, cv_train_labels)

            for n_estimators in [10, 25, 50]:
                clf_dummy = KGPForest(path_max_depth=8)
                clf_dummy.estimators_ = clf.estimators_[:n_estimators]
                preds = clf_dummy.predict(kg, cv_test_entities)

                accuracies[n_estimators].append(accuracy_score(cv_test_labels, preds))

        for n_estimators in [10, 25, 50]:
            avg_acc = np.mean(accuracies[n_estimators])
            std_acc = np.std(accuracies[n_estimators])
        
            if (avg_acc, -std_acc) > best_score:
                best_score = (avg_acc, -std_acc)
                param_combination['n_estimators'] = n_estimators
                best_params = param_combination


    print('Tuned params = {}'.format(best_params))

    N_SPLITS = 5

    # Fit using the tuned parameters
    clf = KGPForest(path_max_depth=8, n_jobs=-1, **best_params)
    
    start = time.time()
    clf.fit(kg, train_entities, train_labels)
    forest_fit_time = time.time() - start

    preds = clf.predict(kg, test_entities)

    print('[Forest] Test accuracy = {} || Confusion Matrix:'.format(accuracy_score(test_labels, preds)))
    print(confusion_matrix(test_labels, preds))

    results['forest_params'] = best_params
    results['forest_fit_time'] = forest_fit_time
    results['forest_preds'] = preds

    # Tune the max_tree_depth
    best_depth, best_score = None, (0, 0)
    for depth in [3, 5, 10, None]:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        accuracies = []
        for train_ix, test_ix in cv.split(train_entities, train_labels):
            cv_train_entities = [train_entities[ix] for ix in train_ix]
            cv_train_labels = [train_labels[ix] for ix in train_ix]
            cv_test_entities = [train_entities[ix] for ix in test_ix]
            cv_test_labels = [train_labels[ix] for ix in test_ix]

            clf = KGPTree(path_max_depth=8, max_tree_depth=depth, n_jobs=-1)
            clf.fit(kg, cv_train_entities, cv_train_labels)
            preds = clf.predict(kg, cv_test_entities)

            accuracies.append(accuracy_score(cv_test_labels, preds))

            ub_accuracies = accuracies + [1.0] * (N_SPLITS - len(accuracies))
            if np.mean(ub_accuracies) < best_score[0]:
                break

        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        if (avg_acc, -std_acc) > best_score:
            best_score = (avg_acc, -std_acc)
            best_depth = depth

    print('Tuned depth = {}'.format(best_depth))

    # Fit using the tuned depth
    clf = KGPTree(path_max_depth=8, max_tree_depth=best_depth, min_samples_leaf=1, n_jobs=-1)
    
    start = time.time()
    clf.fit(kg, train_entities, train_labels)
    tree_fit_time = time.time() - start

    preds = clf.predict(kg, test_entities)

    print('[Tree] Test accuracy = {} || Confusion Matrix:'.format(accuracy_score(test_labels, preds)))
    print(confusion_matrix(test_labels, preds))

    results['tree_depth'] = best_depth
    results['tree_fit_time'] = tree_fit_time
    results['tree_preds'] = preds

    output_file = '{}_{}.p'.format(output[:-2], int(time.time()))
    pickle.dump(results, open(output_file, 'wb+'))

for _ in range(10):

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
    output = 'output/mutag.p'
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
    output = 'output/bgs.p'
    train_model(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, output)

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
    output = 'output/aifb.p'
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
    output = 'output/am.p'
    train_model(rdf_file, format, train_file, test_file, entity_col, label_col, label_predicates, output)

