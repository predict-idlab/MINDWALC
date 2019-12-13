import datastructures as ds
from sklearn.base import ClassifierMixin, TransformerMixin, BaseEstimator
from collections import Counter
import copy
import numpy as np
import itertools
from joblib import Parallel, delayed
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from scipy.stats import entropy
import time
import ray
import psutil

@ray.remote
def _calculate_igs(neighborhoods, labels, walks, n_walks):
    prior_entropy = entropy(np.unique(labels, return_counts=True)[1])
    
    if n_walks > 1:
        top_walks = ds.TopQueue(n_walks)
    else:
        max_ig, top_walk = 0, (None, None)

    for (vertex, depth) in walks:
        features = {0: [], 1: []}
        for inst, label in zip(neighborhoods, labels):
            features[int(inst.find_walk(vertex, depth))].append(label)

        pos_frac = len(features[1]) / len(neighborhoods)
        pos_entr = entropy(np.unique(features[1], return_counts=True)[1])
        neg_frac = len(features[0]) / len(neighborhoods)
        neg_entr = entropy(np.unique(features[0], return_counts=True)[1])
        ig = prior_entropy - (pos_frac * pos_entr + neg_frac * neg_entr)
        if n_walks > 1:
            top_walks.add((vertex, depth), ig)
        else:
            if ig >= max_ig:
                max_ig = ig
                top_walk = (vertex, depth)

    if n_walks > 1:
        return top_walks.data
    else:
        return [(max_ig, top_walk)]

class KGPMixin():
    def __init__(self, path_max_depth=8, progress=None, n_jobs=1, init=True):
        if init:
            if n_jobs == -1:
                n_jobs = psutil.cpu_count(logical=False)
            ray.shutdown()
            ray.init(num_cpus=n_jobs, ignore_reinit_error=True)
        self.path_max_depth = path_max_depth
        self.progress = progress
        self.n_jobs = n_jobs

    def _generate_candidates(self, neighborhoods, sample_frac=None, 
                             useless=None):
        """Generates an iterable with all possible walk candidates."""
        # Generate a set of all possible (vertex, depth) combinations
        walks = set()
        for d in range(2, self.path_max_depth + 1, 2):
            for neighborhood in neighborhoods:
                for vertex in neighborhood.depth_map[d]:
                    walks.add((vertex, d))

        # Prune the useless ones if provided
        if useless is not None:
            old_len = len(walks)
            walks = walks - useless

        # Convert to list so we can sample & shuffle
        walks = list(walks)

        # Sample if sample_frac is provided
        if sample_frac is not None:
            walks_ix = np.random.choice(range(len(walks)), replace=False,
                                        size=int(sample_frac * len(walks)))
            walks = [walks[i] for i in walks_ix]

        # Shuffle the walks (introduces stochastic behaviour to cut ties
        # with similar information gains)
        np.random.shuffle(walks)

        return walks

    def _feature_map(self, walk, neighborhoods, labels):
        """Create two lists of labels of neighborhoods for which the provided
        walk can be found, and a list of labels of neighborhoods for which 
        the provided walk cannot be found."""
        features = {0: [], 1: []}
        vertex, depth = walk
        for i, (inst, label) in enumerate(zip(neighborhoods, labels)):
            features[int(inst.find_walk(vertex, depth))].append(label)
        return features


    def _mine_walks(self, neighborhoods, labels, n_walks=1, sample_frac=None,
                    useless=None):
        """Mine the top-`n_walks` walks that have maximal information gain."""
        walk_iterator = self._generate_candidates(neighborhoods, 
                                                  sample_frac=sample_frac, 
                                                  useless=useless)

        neighborhoods_id = ray.put(neighborhoods)
        labels_id = ray.put(labels)
        walks_id = ray.put(walk_iterator)
        chunk_size = int(np.ceil(len(walk_iterator) / self.n_jobs))

        results = ray.get(
            [_calculate_igs.remote(neighborhoods_id, labels_id, 
                                   walk_iterator[i*chunk_size:(i+1)*chunk_size],
                                   n_walks) 
             for i in range(self.n_jobs)]
        )

        if n_walks > 1:
            top_walks = ds.TopQueue(n_walks)
        else:
            max_ig, top_walk = 0, None

        for data in results:
            for ig, (vertex, depth) in data:
                if n_walks > 1:
                    top_walks.add((vertex, depth), ig)
                else:
                    if ig >= max_ig:
                        max_ig = ig
                        top_walk = (vertex, depth)

        if n_walks > 1:
            return top_walks.data
        else:
            return [(max_ig, top_walk)]

    def _prune_useless(self, neighborhoods, labels):
        """Provide a set of walks that can either be found in all 
        neighborhoods or 1 or less neighborhoods."""
        useless = set()
        walk_iterator = self._generate_candidates(neighborhoods)
        for (vertex, depth) in walk_iterator:
            features = self._feature_map((vertex, depth), neighborhoods, labels)
            if len(features[1]) <= 1 or len(features[1]) == len(neighborhoods):
                useless.add((vertex, depth))
        return useless

    def fit(self, kg, instances, labels):
        if self.progress is not None:
            inst_it = self.progress(instances, desc='Neighborhood extraction')
        else:
            inst_it = instances

        d = self.path_max_depth + 1
        self.neighborhoods = []
        for inst in inst_it:
            neighborhood = kg.extract_neighborhood(inst, d)
            self.neighborhoods.append(neighborhood)


class KGPTree(BaseEstimator, ClassifierMixin, KGPMixin):
    def __init__(self, path_max_depth=8, min_samples_leaf=1, 
                 progress=None, max_tree_depth=None, n_jobs=1,
                 init=True):
        super().__init__(path_max_depth, progress, n_jobs, init)
        self.min_samples_leaf = min_samples_leaf
        self.max_tree_depth = max_tree_depth

    def _stop_condition(self, neighborhoods, labels, curr_tree_depth):
        return (len(set(labels)) == 1 
                or len(neighborhoods) <= self.min_samples_leaf 
                or (self.max_tree_depth is not None 
                    and curr_tree_depth >= self.max_tree_depth))

    def _build_tree(self, neighborhoods, labels, curr_tree_depth=0, 
                    vertex_sample=None, useless=None):

        majority_class = Counter(labels).most_common(1)[0][0]
        if self._stop_condition(neighborhoods, labels, curr_tree_depth):
            return ds.Tree(walk=None, _class=majority_class)

        walks = self._mine_walks(neighborhoods, labels, 
                                 sample_frac=vertex_sample, 
                                 useless=useless)

        if len(walks) == 0 or walks[0][0] == 0:
            return ds.Tree(walk=None, _class=majority_class)

        best_ig, best_walk = walks[0]
        best_vertex, best_depth = best_walk

        node = ds.Tree(walk=best_walk, _class=None)

        found_neighborhoods, found_labels = [], []
        not_found_neighborhoods, not_found_labels = [], []
        
        for neighborhood, label in zip(neighborhoods, labels):
            if neighborhood.find_walk(best_vertex, best_depth):
                found_neighborhoods.append(neighborhood)
                found_labels.append(label)
            else:
                not_found_neighborhoods.append(neighborhood)
                not_found_labels.append(label)
            
        node.right = self._build_tree(found_neighborhoods, found_labels, 
                                      curr_tree_depth=curr_tree_depth + 1,
                                      vertex_sample=vertex_sample,
                                      useless=useless)
            
        node.left = self._build_tree(not_found_neighborhoods, not_found_labels, 
                                     curr_tree_depth=curr_tree_depth + 1,
                                     vertex_sample=vertex_sample,
                                     useless=useless)
        
        return node

    def fit(self, kg, instances, labels):
        super().fit(kg, instances, labels)
        useless = self._prune_useless(self.neighborhoods, labels)
        self.tree_ = self._build_tree(self.neighborhoods, labels, 
                                      useless=useless)

    def predict(self, kg, instances):
        preds = []
        d = self.path_max_depth + 1
        for inst in instances:
            neighborhood = kg.extract_neighborhood(inst, d)
            preds.append(self.tree_.evaluate(neighborhood))
        return preds


class KGPForest(BaseEstimator, ClassifierMixin, KGPMixin):
    def __init__(self, path_max_depth=1, min_samples_leaf=1, 
                 max_tree_depth=None, n_estimators=10, bootstrap=True, 
                 vertex_sample=0.9, progress=None, n_jobs=1):
        super().__init__(path_max_depth, progress, n_jobs)
        self.min_samples_leaf = min_samples_leaf
        self.max_tree_depth = max_tree_depth
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.vertex_sample = vertex_sample

    def _create_estimator(self):
        np.random.seed()

        # Bootstrap the instances if required
        if self.bootstrap:
            sampled_inst_ix = np.random.choice(
                list(range(len(self.neighborhoods))),
                size=len(self.neighborhoods),
                replace=True
            )
            sampled_inst = [self.neighborhoods[i] for i in sampled_inst_ix]
            sampled_labels = [self.labels[i] for i in sampled_inst_ix]
        else:
            sampled_inst = self.neighborhoods
            sampled_labels = self.labels

        # Create a KGPTree, fit it and add to self.estimators_
        tree = KGPTree(self.path_max_depth, self.min_samples_leaf, 
                       self.progress, self.max_tree_depth, self.n_jobs,
                       init=False)
        tree.tree_ = tree._build_tree(sampled_inst, sampled_labels, 
                                      vertex_sample=self.vertex_sample,
                                      useless=self.useless)
        return tree

    def fit(self, kg, instances, labels):
        
        super().fit(kg, instances, labels)
        useless = self._prune_useless(self.neighborhoods, labels)

        self.labels = labels
        self.useless = useless

        if self.progress is not None and self.n_jobs == 1:
            estimator_iterator = self.progress(range(self.n_estimators), 
                                               desc='estimator loop', 
                                               leave=True)
        else:
            estimator_iterator = range(self.n_estimators)

        self.estimators_ = []
        for _ in estimator_iterator:
            self.estimators_.append(self._create_estimator())

    def predict(self, kg, instances):
        if self.progress is not None:
            inst_iterator = self.progress(instances, 
                                          desc='Extracting neighborhoods')
        else:
            inst_iterator = instances

        neighborhoods = []
        d = self.path_max_depth + 1
        for inst in inst_iterator:
            neighborhood = kg.extract_neighborhood(inst, depth=d)
            neighborhoods.append(neighborhood)

        predictions = []
        for neighborhood in neighborhoods:
            inst_preds = []
            for tree in self.estimators_:
                inst_preds.append(tree.tree_.evaluate(neighborhood))
            predictions.append(Counter(inst_preds).most_common()[0][0])
        return predictions

class KPGTransformer(BaseEstimator, TransformerMixin, KGPMixin):
    def __init__(self, path_max_depth=8, progress=None, n_jobs=1, 
                 n_features=1):
        super().__init__(path_max_depth, progress, n_jobs)
        self.n_features = n_features

    def fit(self, kg, instances, labels):
        if self.progress is not None:
            inst_iterator = self.progress(instances, 
                                          desc='Extracting neighborhoods')
        else:
            inst_iterator = instances

        neighborhoods = []
        d = self.path_max_depth + 1
        for inst in inst_iterator:
            neighborhood = kg.extract_neighborhood(inst, depth=d)
            neighborhoods.append(neighborhood)

        prior_entropy = entropy(np.unique(labels, return_counts=True)[1])

        cache = {}

        self.walks_ = set()

        if len(np.unique(labels)) > 2:
            _classes = np.unique(labels)
        else:
            _classes = [labels[0]]

        for _class in _classes:
            label_map = {}
            for lab in np.unique(labels):
                if lab == _class:
                    label_map[lab] = 1
                else:
                    label_map[lab] = 0

            new_labels = list(map(lambda x: label_map[x], labels))

            walks = self._mine_walks(neighborhoods, new_labels, 
                                     n_walks=self.n_features)

            prev_len = len(self.walks_)
            n_walks = min(self.n_features // len(np.unique(labels)), len(walks))
            for _, walk in sorted(walks, key=lambda x: -x[0]):
                if len(self.walks_) - prev_len >= n_walks:
                    break

                if walk not in self.walks_:
                    self.walks_.add(walk)

    def transform(self, kg, instances):
        if self.progress is not None:
            inst_iterator = self.progress(instances, 
                                          desc='Extracting neighborhoods')
        else:
            inst_iterator = instances

        neighborhoods = []
        d = self.path_max_depth + 1
        for inst in inst_iterator:
            neighborhood = kg.extract_neighborhood(inst, depth=d)
            neighborhoods.append(neighborhood)

        features = np.zeros((len(instances), self.n_features))
        for i, neighborhood in enumerate(neighborhoods):
            for j, (vertex, depth) in enumerate(self.walks_):
                features[i, j] = neighborhood.find_walk(vertex, depth)
        return features