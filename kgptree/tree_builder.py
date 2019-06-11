from datastructures import *
from sklearn.base import ClassifierMixin, BaseEstimator
from tqdm import tqdm
from collections import Counter


class KGPTree(BaseEstimator, ClassifierMixin):
    def __init__(self, kg, path_max_depth=1, neighborhood_depth=8, 
                 min_samples_leaf=2, max_tree_depth=None):
        self.kg = kg
        self.path_max_depth = path_max_depth
        self.neighborhood_depth = neighborhood_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_tree_depth = max_tree_depth

    def _create_walk(self, depth, vertex):
        walk = Walk()
        walk.append(Hop('', root=True))
        for _ in range(depth - 1):
            walk.append(Hop(Vertex(''), wildcard=True))
        walk.append(Hop(vertex))
        return walk

    def _build_path(self, neighborhoods, labels):
        prior_entropy = entropy(np.unique(labels, return_counts=True)[1])
        igs = {}
        vertices = set(filter(lambda x: not x.predicate, 
                              set.union(*[neighborhood.vertices 
                                          for neighborhood in neighborhoods])))

        for d in range(2, self.path_max_depth + 1, 2):
            for vertex in tqdm(vertices):
                walk = self._create_walk(d, vertex)
                igs[walk] = walk.calc_ig(self.kg, neighborhoods, labels, 
                                         prior_entropy=prior_entropy)
            
        return sorted(igs.items(), key=lambda x: (-x[1]))[0]

    def _stop_condition(self, neighborhoods, labels, curr_tree_depth):
        return (len(set(labels)) == 1 
                or len(neighborhoods) <= self.min_samples_leaf 
                or (self.max_tree_depth is not None 
                and curr_tree_depth >= self.max_tree_depth))


    def build_tree(self, neighborhoods, labels, curr_tree_depth=0):
        # Before doing many calculations, check if the stop conditions are met
        if self._stop_condition(neighborhoods, labels, curr_tree_depth):
            return Tree(walk=None, _class=Counter(labels).most_common(1)[0][0])
        
        # Create the path that maximizes information gain for these neighborhoods
        best_walk, best_ig = self._build_path(neighborhoods, labels)
        
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
                                     curr_tree_depth=curr_tree_depth + 1)
            
        node.left = self.build_tree(not_found_neighborhoods, not_found_labels, 
                                    curr_tree_depth=curr_tree_depth + 1)
        
        return node


    def fit(self, instances, labels, curr_tree_depth=0):
        # TODO: Check if input is alright

        print('Extracting neighborhoods...')
        neighborhoods = []
        for inst in tqdm(instances):
            neighborhood = self.kg.extract_instance(inst, depth=self.neighborhood_depth)
            neighborhoods.append(neighborhood)

        self.tree_ = self.build_tree(neighborhoods, labels)


    def predict(self, instances):
        # TODO: Check if model has been fitted

        return [self.tree_.evaluate(self.kg.extract_instance(inst, depth=self.neighborhood_depth), self.kg) for inst in instances]