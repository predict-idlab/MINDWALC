from datastructures import *
from sklearn.base import ClassifierMixin, BaseEstimator
from collections import Counter
import copy


class KGPTree(BaseEstimator, ClassifierMixin):
    def __init__(self, kg, path_max_depth=8, min_samples_leaf=2, 
                 progress=None, max_tree_depth=None):
        self.kg = kg
        self.path_max_depth = path_max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_tree_depth = max_tree_depth
        self.progress = progress

    def _create_walk(self, depth, vertex):
        walk = Walk()
        walk.append(Hop('', root=True))
        for _ in range(depth - 1):
            walk.append(Hop(Vertex(''), wildcard=True))
        walk.append(Hop(vertex))
        return walk

    def _build_path(self, neighborhoods, labels, allowed_v=None):
        prior_entropy = entropy(np.unique(labels, return_counts=True)[1])
        igs = {}
        vertices = set(filter(lambda x: not x.predicate, 
                              set.union(*[neighborhood.vertices 
                                          for neighborhood in neighborhoods])))
        if allowed_v is not None:
            vertices = vertices.intersection(allowed_v)

        vertices = list(vertices)

        # Sort them for determinism
        # vertices = sorted(vertices)

        # Randomly permute the order (introduces stochastic behaviour)
        np.random.shuffle(vertices)

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
                igs[walk] = walk.calc_ig(self.kg, neighborhoods, labels, 
                                         prior_entropy=prior_entropy)
            
        return sorted(igs.items(), key=lambda x: (-x[1]))[0]

    def _stop_condition(self, neighborhoods, labels, curr_tree_depth):
        return (len(set(labels)) == 1 
                or len(neighborhoods) <= self.min_samples_leaf 
                or (self.max_tree_depth is not None 
                    and curr_tree_depth >= self.max_tree_depth))


    def build_tree(self, neighborhoods, labels, curr_tree_depth=0, 
                   allowed_v=None):
        # Before doing many calculations, check if the stop conditions are met
        if self._stop_condition(neighborhoods, labels, curr_tree_depth):
            return Tree(walk=None, _class=Counter(labels).most_common(1)[0][0])
        
        # Create the path that maximizes information gain for these neighborhoods
        best_walk, best_ig = self._build_path(neighborhoods, labels, 
                                              allowed_v=allowed_v)
        
        if best_ig == 0:
            return Tree(walk=None, _class=Counter(labels).most_common(1)[0][0])
        
        # Create the node in our tree, partition the data and continue recursively
        node = Tree(walk=best_walk, _class=None)
        
        found_neighborhoods, found_labels = [], []
        not_found_neighborhoods, not_found_labels = [], []
        
        for neighborhood, label in zip(neighborhoods, labels):
            if neighborhood.find_walk(best_walk, self.kg):
                found_neighborhoods.append(neighborhood)
                found_labels.append(label)
            else:
                not_found_neighborhoods.append(neighborhood)
                not_found_labels.append(label)
            
        node.right = self.build_tree(found_neighborhoods, found_labels, 
                                     curr_tree_depth=curr_tree_depth + 1,
                                     allowed_v=allowed_v)
            
        node.left = self.build_tree(not_found_neighborhoods, not_found_labels, 
                                    curr_tree_depth=curr_tree_depth + 1,
                                    allowed_v=allowed_v)
        
        return node


    def fit(self, instances, labels):
        # TODO: Check if input is alright

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

        self.tree_ = self.build_tree(neighborhoods, labels)


    def predict(self, instances):
        # TODO: Check if model has been fitted

        return [self.tree_.evaluate(self.kg.extract_instance(inst, depth=self.path_max_depth), self.kg) for inst in instances]


class KGPForest(BaseEstimator, ClassifierMixin):
    def __init__(self, kg, path_max_depth=1, min_samples_leaf=2, 
                 max_tree_depth=None, n_estimators=10, bootstrap=True, 
                 vertex_sample=0.9, progress=False):
        self.kg = kg
        self.path_max_depth = path_max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_tree_depth = max_tree_depth
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.vertex_sample = vertex_sample
        self.progress = progress

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

        if self.progress is not None:
            estimator_iterator = self.progress(range(self.n_estimators), 
                                               desc='estimator loop', 
                                               leave=True)
        else:
            estimator_iterator = range(self.n_estimators)

        self.estimators_ = []
        for _ in estimator_iterator:
            # Bootstrap the instances if required
            if self.bootstrap:
                sampled_inst_ix = np.random.choice(
                    list(range(len(instances))),
                    size=len(instances),
                    replace=True
                )
                sampled_inst = [neighborhoods[i] for i in sampled_inst_ix]
                sampled_labels = [labels[i] for i in sampled_inst_ix]
            else:
                sampled_inst = neighborhoods
                sampled_labels = labels

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
                                         allowed_v=sampled_vertices)
            self.estimators_.append(tree)

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
            print(Counter(inst_preds))
            predictions.append(Counter(inst_preds).most_common()[0][0])
        return predictions