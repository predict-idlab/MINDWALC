from datastructures import *
from sklearn.base import ClassifierMixin, TransformerMixin, BaseEstimator
from collections import Counter
import copy
from joblib import Parallel, delayed
from multiprocessing import Pool


class KGPMixin():
    def __init__(self, kg, path_max_depth=8, progress=None, n_jobs=1):
        self.kg = kg
        self.path_max_depth = path_max_depth
        self.progress = progress
        self.n_jobs = n_jobs

    def _create_walk(self, depth, vertex):
        walk = Walk()
        walk.append(Hop('', root=True))
        for _ in range(depth - 1):
            walk.append(Hop(Vertex(''), wildcard=True))
        walk.append(Hop(vertex))
        return walk

    def _get_useless(self, neighborhoods, labels):
        useless = []

        vertices = set(filter(lambda x: not x.predicate, 
                              set.union(*[neighborhood.vertices 
                                          for neighborhood in neighborhoods])))

        vertices = list(vertices)

        if self.progress is not None:
            depth_iterator = self.progress(range(2, self.path_max_depth + 1, 2),
                                           desc='depth loop', 
                                           leave=False)
        else:
            depth_iterator = range(2, self.path_max_depth + 1, 2)

        for d in depth_iterator:
        # for d in range(self.path_max_depth + 1):
            if self.progress is not None:
                vertex_iterator = self.progress(vertices, desc='vertex loop', 
                                                leave=False)
            else:
                vertex_iterator = vertices

            for vertex in vertex_iterator:
                walk = self._create_walk(d, vertex)

                n_found = 0
                for neighborhood, label in zip(neighborhoods, labels):
                    if neighborhood.find_walk(walk):
                        n_found += 1

                if n_found <= 1 or n_found == len(neighborhoods):
                    useless.append((len(walk) - 1, walk[-1].vertex.get_name()))

        return set(useless)


class KGPTree(BaseEstimator, ClassifierMixin, KGPMixin):
    def __init__(self, kg, path_max_depth=8, min_samples_leaf=2, 
                 progress=None, max_tree_depth=None, n_jobs=1):
        super().__init__(kg, path_max_depth, progress, n_jobs)
        self.min_samples_leaf = min_samples_leaf
        self.max_tree_depth = max_tree_depth

    def _check_candidate(self, v, d, neighborhoods, labels, igs, 
                         prior_entropy, find_walk_cache, useless):
        if useless is not None and (d, v.get_name()) in useless:
            return
        walk = self._create_walk(d, v)
        igs[walk] = walk.calc_ig(self.kg, neighborhoods, labels, 
                                 prior_entropy=prior_entropy,
                                 cache=find_walk_cache)

    def _build_path(self, neighborhoods, labels, allowed_v=None, 
                    find_walk_cache=None, useless=None):
        prior_entropy = entropy(np.unique(labels, return_counts=True)[1])
        igs = {}
        vertices = set(filter(lambda x: not x.predicate, 
                              set.union(*[neighborhood.vertices 
                                          for neighborhood in neighborhoods])))
        if allowed_v is not None:
            vertices = vertices.intersection(allowed_v)

        vertices = list(vertices)

        # Randomly permute the order
        np.random.shuffle(vertices)

        if self.progress is not None and self.n_jobs == 1:
            depth_iterator = self.progress(range(2, self.path_max_depth + 1, 2),
                                           desc='depth loop', 
                                           leave=False)
        else:
            depth_iterator = range(2, self.path_max_depth + 1, 2)

        for d in depth_iterator:
            if self.progress is not None and self.n_jobs == 1:
                vertex_iterator = self.progress(vertices, desc='vertex loop', 
                                                leave=False)
            else:
                vertex_iterator = vertices

            for vertex in vertex_iterator:
                self._check_candidate(vertex, d, neighborhoods, labels, 
                                      igs, prior_entropy, find_walk_cache,
                                      useless)
            
        return max(igs.items(), key=lambda x: x[1])

    def _stop_condition(self, neighborhoods, labels, curr_tree_depth):
        return (len(set(labels)) == 1 
                or len(neighborhoods) <= self.min_samples_leaf 
                or (self.max_tree_depth is not None 
                    and curr_tree_depth >= self.max_tree_depth))


    def build_tree(self, neighborhoods, labels, curr_tree_depth=0, 
                   allowed_v=None, find_walk_cache=None, useless=None):
        # Before doing many calculations, check if the stop conditions are met
        if self._stop_condition(neighborhoods, labels, curr_tree_depth):
            return Tree(walk=None, _class=Counter(labels).most_common(1)[0][0])
        
        # Create the path that maximizes information gain for these neighborhoods
        best_walk, best_ig = self._build_path(neighborhoods, labels, 
                                              allowed_v=allowed_v,
                                              find_walk_cache=find_walk_cache,
                                              useless=useless)
        
        if best_ig == 0:
            return Tree(walk=None, _class=Counter(labels).most_common(1)[0][0])
        
        # Create the node in our tree, partition the data and continue recursively
        node = Tree(walk=best_walk, _class=None)
        
        found_neighborhoods, found_labels = [], []
        not_found_neighborhoods, not_found_labels = [], []
        
        for neighborhood, label in zip(neighborhoods, labels):
            if neighborhood.find_walk(best_walk):
                found_neighborhoods.append(neighborhood)
                found_labels.append(label)
            else:
                not_found_neighborhoods.append(neighborhood)
                not_found_labels.append(label)
            
        node.right = self.build_tree(found_neighborhoods, found_labels, 
                                     curr_tree_depth=curr_tree_depth + 1,
                                     allowed_v=allowed_v,
                                     find_walk_cache=find_walk_cache,
                                     useless=useless)
            
        node.left = self.build_tree(not_found_neighborhoods, not_found_labels, 
                                    curr_tree_depth=curr_tree_depth + 1,
                                    allowed_v=allowed_v,
                                    find_walk_cache=find_walk_cache,
                                    useless=useless)
        
        return node


    def fit(self, instances, labels):
        if self.progress is not None:
            inst_iterator = self.progress(instances, 
                                          desc='Extracting neighborhoods')
        else:
            inst_iterator = instances

        neighborhoods = []
        for inst in inst_iterator:
            neighborhood = self.kg.extract_instance(inst, 
                                                    depth=self.path_max_depth)
            neighborhoods.append(neighborhood)

        useless = self._get_useless(neighborhoods, labels)

        self.tree_ = self.build_tree(neighborhoods, labels, 
                                     find_walk_cache={},
                                     useless=useless)


    def predict(self, instances):
        return [self.tree_.evaluate(self.kg.extract_instance(inst, depth=self.path_max_depth), self.kg) for inst in instances]


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

        # Sample in the vertices
        sampled_vertices = np.random.choice(
            list(self.kg.vertices),
            size=int(self.vertex_sample * len(self.kg.vertices)),
            replace=False
        )

        # Create a KGPTree, fit it and add to self.estimators_
        tree = KGPTree(self.kg, self.path_max_depth, 
                       self.min_samples_leaf,
                       self.progress,
                       self.max_tree_depth)
        tree.tree_ = tree.build_tree(sampled_inst, sampled_labels, 
                                     allowed_v=sampled_vertices,
                                     find_walk_cache=self.walk_cache,
                                     useless=self.useless)
        return tree

    def fit(self, instances, labels):
        if self.progress is not None:
            inst_iterator = self.progress(instances, 
                                          desc='Extracting neighborhoods')
        else:
            inst_iterator = instances

        neighborhoods = []
        for inst in inst_iterator:
            neighborhood = self.kg.extract_instance(inst, depth=self.path_max_depth)
            neighborhoods.append(neighborhood)

        if self.progress is not None and self.n_jobs == 1:
            estimator_iterator = self.progress(range(self.n_estimators), 
                                               desc='estimator loop', 
                                               leave=True)
        else:
            estimator_iterator = range(self.n_estimators)

        walk_cache = {}
        useless = self._get_useless(neighborhoods, labels)

        self.neighborhoods = neighborhoods
        self.labels = labels
        self.walk_cache = walk_cache
        self.useless = useless

        self.estimators_ = []
        if self.n_jobs == 1:
            for _ in estimator_iterator:
                self.estimators_.append(self._create_estimator())
        else:
            p = Pool(self.n_jobs)
            self.estimators_ = p.map(self._create_estimator, estimator_iterator)

    def predict(self, instances):
        if self.progress is not None:
            inst_iterator = self.progress(instances, 
                                          desc='Extracting neighborhoods')
        else:
            inst_iterator = instances

        neighborhoods = []
        for inst in inst_iterator:
            neighborhood = self.kg.extract_instance(inst, depth=self.path_max_depth)
            neighborhoods.append(neighborhood)

        predictions = []
        for neighborhood in neighborhoods:
            inst_preds = []
            for tree in self.estimators_:
                inst_preds.append(tree.tree_.evaluate(neighborhood, self.kg))
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
        for inst in inst_iterator:
            neighborhood = self.kg.extract_instance(inst, depth=self.path_max_depth)
            neighborhoods.append(neighborhood)

        prior_entropy = entropy(np.unique(labels, return_counts=True)[1])
        vertices = set(filter(lambda x: not x.predicate, 
                              set.union(*[neighborhood.vertices 
                                          for neighborhood in neighborhoods])))

        vertices = list(vertices)
        cache = {}

        # Randomly permute the order (introduces stochastic behaviour)
        np.random.shuffle(vertices)

        self.walks_ = set()

        if len(np.unique(labels)) > 2:
            _classes = np.unique(labels)
        else:
            _classes = [labels[0]]

        for _class in _classes:
            igs = {}
            label_map = {}
            for lab in np.unique(labels):
                if lab == _class:
                    label_map[lab] = 1
                else:
                    label_map[lab] = 0

            new_labels = list(map(lambda x: label_map[x], labels))

            if self.progress is not None:
                depth_iterator = self.progress(range(2, self.path_max_depth + 1, 2),
                                               desc='depth loop', 
                                               leave=False)
            else:
                depth_iterator = range(2, self.path_max_depth + 1, 2)

            for d in depth_iterator:
                if self.progress is not None:
                    vertex_iterator = self.progress(vertices, desc='vertex loop', 
                                                    leave=False)
                else:
                    vertex_iterator = vertices

                for vertex in vertex_iterator:
                    walk = self._create_walk(d, vertex)
                    igs[walk] = walk.calc_ig(self.kg, neighborhoods, new_labels, 
                                             prior_entropy=prior_entropy,
                                             cache=cache)

            prev_len = len(self.walks_)
            n_walks = min(self.n_features // len(np.unique(labels)), len(igs))
            for walk, _ in sorted(igs.items(), key=lambda x: -x[1]):
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
        for inst in inst_iterator:
            neighborhood = self.kg.extract_instance(inst, depth=self.path_max_depth)
            neighborhoods.append(neighborhood)

        features = np.zeros((len(instances), self.n_features))
        for i, neighborhood in enumerate(neighborhoods):
            for j, walk in enumerate(self.walks_):
                features[i, j] = neighborhood.find_walk(walk)
        return features