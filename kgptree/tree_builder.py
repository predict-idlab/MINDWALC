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


class KGPMixin():
    def __init__(self, kg, path_max_depth=8, progress=None, n_jobs=1):
        self.kg = kg
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
                    if not vertex.predicate:
                        walks.add(ds.Walk(vertex, d))

        # Prune the useless ones if provided
        if useless is not None:
            walks = walks - useless

        # Convert to list so we can sample & shuffle
        walks = list(walks)

        # Sample if sample_frac is provided
        if sample_frac is not None:
            walks = np.random.choice(walks, replace=False,
                                     size=int(sample_frac * len(walks)))

        # Shuffle the walks (introduces stochastic behaviour to cut ties
        # with similar information gains)
        np.random.shuffle(walks)

        # Wrap in a tqdm progressbar if is provided
        if self.progress is not None and self.n_jobs == 1:
            walk_iterator = self.progress(walks, desc='walk loop', 
                                          leave=False)
        else:
            walk_iterator = walks

        return walk_iterator

    def _feature_map(self, walk, neighborhoods, labels):
        """Create two lists of labels of neighborhoods for which the provided
        walk can be found, and a list of labels of neighborhoods for which 
        the provided walk cannot be found."""
        features = {0: [], 1: []}
        for i, (inst, label) in enumerate(zip(neighborhoods, labels)):
            features[int(inst.find_walk(walk))].append(label)
        return features

    def _mine_walks(self, neighborhoods, labels, n_walks=1, sample_frac=None,
                    useless=None):
        """Mine the top-`n_walks` walks that have maximal information gain."""
        walk_iterator = self._generate_candidates(neighborhoods, 
                                                  sample_frac=sample_frac, 
                                                  useless=useless)

        prior_entropy = entropy(np.unique(labels, return_counts=True)[1])
        top_walks = ds.TopQueue(n_walks)
        for walk in walk_iterator:
            features = self._feature_map(walk, neighborhoods, labels)
            pos_frac = len(features[1]) / len(neighborhoods)
            pos_entr = entropy(np.unique(features[1], return_counts=True)[1])
            neg_frac = len(features[0]) / len(neighborhoods)
            neg_entr = entropy(np.unique(features[0], return_counts=True)[1])
            ig = prior_entropy - (pos_frac * pos_entr + neg_frac * neg_entr)
            top_walks.add(walk, ig)

        return top_walks.data

    def _prune_useless(self, neighborhoods, labels):
        """Provide a set of walks that can either be found in all 
        neighborhoods or 1 or less neighborhoods."""
        useless = set()
        walk_iterator = self._generate_candidates(neighborhoods)
        for walk in walk_iterator:
            features = self._feature_map(walk, neighborhoods, labels)
            if len(features[1]) <= 1 or len(features[1]) == len(neighborhoods):
                useless.add(walk)
        return useless

    def fit(self, instances, labels):
        if self.progress is not None:
            inst_it = self.progress(instances, desc='Neighborhood extraction')
        else:
            inst_it = instances

        d = self.path_max_depth + 1
        self.neighborhoods = []
        for inst in inst_it:
            neighborhood = self.kg.extract_neighborhood(inst, d)
            self.neighborhoods.append(neighborhood)


class KGPTree(BaseEstimator, ClassifierMixin, KGPMixin):
    def __init__(self, kg, path_max_depth=8, min_samples_leaf=2, 
                 progress=None, max_tree_depth=None, n_jobs=1):
        super().__init__(kg, path_max_depth, progress, n_jobs)
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
        
        node = ds.Tree(walk=best_walk, _class=None)

        found_neighborhoods, found_labels = [], []
        not_found_neighborhoods, not_found_labels = [], []
        
        for neighborhood, label in zip(neighborhoods, labels):
            if neighborhood.find_walk(best_walk):
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

    def fit(self, instances, labels):
        super().fit(instances, labels)
        useless = self._prune_useless(self.neighborhoods, labels)
        self.tree_ = self._build_tree(self.neighborhoods, labels, 
                                      useless=useless)

    def predict(self, instances):
        preds = []
        d = self.path_max_depth + 1
        for inst in instances:
            neighborhood = self.kg.extract_neighborhood(inst, d)
            preds.append(self.tree_.evaluate(neighborhood))
        return preds


class KGPForest(BaseEstimator, ClassifierMixin, KGPMixin):
    def __init__(self, kg, path_max_depth=1, min_samples_leaf=2, 
                 max_tree_depth=None, n_estimators=10, bootstrap=True, 
                 vertex_sample=0.9, progress=False, n_jobs=1):
        super().__init__(kg, path_max_depth, progress, n_jobs)
        self.min_samples_leaf = min_samples_leaf
        self.max_tree_depth = max_tree_depth
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.vertex_sample = vertex_sample

    def _create_estimator(self):
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
        tree = KGPTree(self.kg, self.path_max_depth, self.min_samples_leaf, 
                       self.progress, self.max_tree_depth, self.n_jobs)
        tree.tree_ = tree._build_tree(sampled_inst, sampled_labels, 
                                      vertex_sample=self.vertex_sample,
                                      useless=self.useless)
        return tree

    def fit(self, instances, labels):
        
        super().fit(instances, labels)
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
        if self.n_jobs == 1:
            for _ in estimator_iterator:
                self.estimators_.append(self._create_estimator())
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs, backend='multiprocessing')(
                delayed(self._create_estimator)()
                for _ in estimator_iterator
            )

    def predict(self, instances):
        if self.progress is not None:
            inst_iterator = self.progress(instances, 
                                          desc='Extracting neighborhoods')
        else:
            inst_iterator = instances

        neighborhoods = []
        d = self.path_max_depth + 1
        for inst in inst_iterator:
            neighborhood = self.kg.extract_neighborhood(inst, depth=d)
            neighborhoods.append(neighborhood)

        predictions = []
        for neighborhood in neighborhoods:
            inst_preds = []
            for tree in self.estimators_:
                inst_preds.append(tree.tree_.evaluate(neighborhood))
            predictions.append(Counter(inst_preds).most_common()[0][0])
        return predictions

class KPGTransformer(BaseEstimator, TransformerMixin, KGPMixin):
    def __init__(self, kg, path_max_depth=8, progress=None, n_jobs=1, 
                 n_features=1):
        super().__init__(kg, path_max_depth, progress, n_jobs)
        self.n_features = n_features

    def fit(self, instances, labels):
        if self.progress is not None:
            inst_iterator = self.progress(instances, 
                                          desc='Extracting neighborhoods')
        else:
            inst_iterator = instances

        neighborhoods = []
        d = self.path_max_depth + 1
        for inst in inst_iterator:
            neighborhood = self.kg.extract_neighborhood(inst, depth=d)
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

    def transform(self, instances):
        if self.progress is not None:
            inst_iterator = self.progress(instances, 
                                          desc='Extracting neighborhoods')
        else:
            inst_iterator = instances

        neighborhoods = []
        d = self.path_max_depth + 1
        for inst in inst_iterator:
            neighborhood = self.kg.extract_neighborhood(inst, depth=d)
            neighborhoods.append(neighborhood)

        features = np.zeros((len(instances), self.n_features))
        for i, neighborhood in enumerate(neighborhoods):
            for j, walk in enumerate(self.walks_):
                features[i, j] = neighborhood.find_walk(walk)
        return features