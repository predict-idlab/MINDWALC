import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp

from collections import defaultdict, Counter, OrderedDict
from functools import lru_cache

import os
import itertools
import time

import rdflib

from scipy.stats import entropy

# The idea of using a hashing function is taken from:
# https://github.com/benedekrozemberczki/graph2vec
from hashlib import md5
import copy


############################################################################
#                  Vertex & KnowledgeGraph objects                         #
############################################################################

class Vertex(object):
    vertex_counter = 0
    
    def __init__(self, name, predicate=False, _from=None, _to=None, 
                 wildcard=False, literal=False):
        self.name = name
        self.predicate = predicate
        self._from = _from
        self._to = _to
        self.wildcard = wildcard
        self.literal = literal
        
    def __eq__(self, other):
        if other is None: 
            return False
        return self.__hash__() == other.__hash__()
    
    def get_name(self):
        if self.wildcard:
            return '*'
        else:
            return self.name
    
    def __hash__(self):
        if self.predicate:
            return hash((self._from, self._to, self.name))
        else:
            return hash(self.name)

    def __lt__(self, other):
        if self.predicate and not other.predicate:
            return False
        if not self.predicate and other.predicate:
            return True
        if self.predicate:
            return (self.name, self._from, self._to) < (other.name, other._from, other._to)
        else:
            return self.name < other.name

class KnowledgeGraph(object):
    _id = 0
    def __init__(self):
        self.vertices = set()
        self.transition_matrix = defaultdict(set)
        self.inv_transition_matrix = defaultdict(set)
        self.label_map = defaultdict(dict)
        self.inv_label_map = {}
        self.name_to_vertex = {}
        self.root = None
        self._id = KnowledgeGraph._id
        KnowledgeGraph._id += 1
        
    def add_vertex(self, vertex):
        if vertex.predicate:
            self.vertices.add(vertex)
            
        if not vertex.predicate and vertex not in self.vertices:
            self.vertices.add(vertex)

        self.label_map[vertex][0] = vertex.get_name()
        self.name_to_vertex[vertex.name] = vertex

    def add_edge(self, v1, v2):
        # Uni-directional edge
        self.transition_matrix[v1].add(v2)
        self.inv_transition_matrix[v2].add(v1)
        
    def remove_edge(self, v1, v2):
        if v2 in self.transition_matrix[v1]:
            self.transition_matrix[v1].remove(v2)
            self.inv_transition_matrix[v2].remove(v1)

    def get_neighbors(self, vertex):
        return self.transition_matrix[vertex]
    
    def get_neighbors_inv(self, vertex):
        return self.inv_transition_matrix[vertex]
    
    def get_entity_neighbors(self, vertex):
        if vertex.predicate:
            return self.transition_matrix[vertex]
        else:
            neighbors = []
            for neighbor in self.transition_matrix[vertex]:
                neighbors += [(neighbor, x) 
                              for x in self.transition_matrix[neighbor]]
            return neighbors
    
    def get_entity_neighbors_inv(self, vertex):
        if vertex.predicate:
            return self.inv_transition_matrix[vertex]
        else:
            neighbors = []
            for neighbor in self.inv_transition_matrix[vertex]:
                neighbors += [(neighbor, x) 
                              for x in self.inv_transition_matrix[neighbor]]
            return neighbors
    
    def visualise(self):
        nx_graph = nx.DiGraph()
        
        for v in self.vertices:
            if not v.predicate:
                name = v.name.split('/')[-1]
                nx_graph.add_node(name, name=name, pred=v.predicate)
            
        for v in self.vertices:
            if not v.predicate:
                v_name = v.name.split('/')[-1]
                # Neighbors are predicates
                for pred in self.get_neighbors(v):
                    pred_name = pred.name.split('/')[-1]
                    for obj in self.get_neighbors(pred):
                        obj_name = obj.name.split('/')[-1]
                        nx_graph.add_edge(v_name, obj_name, name=pred_name)
        
        plt.figure(figsize=(10,10))
        _pos = nx.circular_layout(nx_graph)
        nx.draw_networkx_nodes(nx_graph, pos=_pos)
        nx.draw_networkx_edges(nx_graph, pos=_pos)
        nx.draw_networkx_labels(nx_graph, pos=_pos)
        nx.draw_networkx_edge_labels(nx_graph, pos=_pos, 
                                     edge_labels=nx.get_edge_attributes(nx_graph, 'name'))
        plt.show()
    
    def _create_label(self, vertex, n):
        neighbor_names = [self.label_map[x][n - 1] for x in self.get_neighbors(vertex)]
        suffix = '-'.join(sorted(set(map(str, neighbor_names))))
        return self.label_map[vertex][n - 1] + '-' + suffix
        
    def weisfeiler_lehman(self, iterations=3):
        # Store the WL labels in a dictionary with a two-level key:
        # First level is the vertex identifier
        # Second level is the WL iteration
        self.label_map = defaultdict(dict)
        self.inv_label_map = {}

        for v in self.vertices:
            self.label_map[v][0] = v.get_name()
            self.inv_label_map[v.name] = v
        
        for n in range(1, iterations+1):
            for vertex in self.vertices:
                if len(self.get_neighbors(vertex)) == 0:
                    self.label_map[vertex][n] = self.label_map[vertex][n - 1]
                
                # Create multi-set label
                s_n = self._create_label(vertex, n)

                # Store it in our label_map (hash trick from: benedekrozemberczki/graph2vec)
                digest = str(md5(s_n.encode()).hexdigest())
                self.label_map[vertex][n] = digest
                self.inv_label_map[digest] = vertex
    
    def find_walk(self, walk):
        return walk[-1].vertex.get_name() in self.depth_map[len(walk) - 1]
        # if len(list(filter(lambda x: not x.wildcard and not x.root, walk))) == 1:
        #     return walk[-1].vertex.get_name() in self.depth_map[len(walk) - 1]
        
        # # Process first element of walk: entity, root or wildcard (TODO: This should always be root!)
        # if walk[0].root:
        #     to_explore = self.get_neighbors(self.root)
        # elif walk[0].wildcard:
        #     print('WARNING: Found a root of a walk which is a wildcard, this should not happen!')
        #     to_explore = [self.get_neighbors(x) for x in self.vertices]
        # else:
        #     if walk[0].vertex not in self.vertices: 
        #         return False
        #     to_explore = self.get_neighbors(walk[0].vertex)
            
        # # Let's first check if all vertices in the walk are present
        # for hop in walk[1:]:
        #     if not hop.wildcard and hop.vertex not in self.vertices:
        #         return False
        
        # # TODO: This code below is currently never executed, since
        # # all walks are of the form root --> wildcards --> vertex

        # # Process second element until end. Alternate between entities and predicates.
        # for hop_nr, hop in enumerate(walk[1:]):
                               
        #     if hop_nr == len(walk) - 1:
        #         return True
                
        #     #print(hop, hop.vertex.name, [x.name for x in to_explore])
        #     new_explore = set()
        #     if not hop.vertex.predicate:  # Entity
        #         if hop.wildcard:
        #             for node in to_explore:
        #                 for neighbor in self.get_neighbors(node):
        #                     new_explore.add(neighbor)
        #         else:
        #             for node in to_explore:
        #                 if hop.vertex.name in [x.name for x in self.get_neighbors(node)]:
        #                     new_explore.add(self.name_to_vertex[hop.vertex.name])
        #     else:  # Predicate
        #         for node in to_explore:
        #             for neighbor in self.get_neighbors(node):
        #                 if hop.wildcard or neighbor.name == hop.vertex.name:
        #                     new_explore.add(neighbor)

        #     to_explore = new_explore
            
        #     if len(to_explore) == 0:
        #         return False


        # return True

    def extract_random_walks(self, depth, max_walks=None, root=None):
        # Initialize one walk of length 1 (the root)
        if root is None:
            walks = {(self.root)}
        else:
            walks = {(root)}

        for i in range(depth):
            # In each iteration, iterate over the walks, grab the 
            # last hop, get all its neighbors and extend the walks
            walks_copy = walks.copy()
            for walk in walks_copy:
                node = walk[-1]
                neighbors = self.get_neighbors(node)

                if len(neighbors) > 0:
                    walks.remove(walk)

                for neighbor in neighbors:
                    walks.add(tuple(list(walk) + [neighbor]))

            # TODO: Should we prune in every iteration?
            if max_walks is not None:
                walks_ix = np.random.choice(range(len(walks)), replace=False, 
                                            size=min(len(walks), max_walks))
                if len(walks_ix) > 0:
                    walks = np.array(walks)[walks_ix].tolist()

        # Return a numpy array of these walks
        return np.array(walks)


    def extract_instance(self, instance, depth=8):
        subgraph = KnowledgeGraph()
        subgraph.label_map = self.label_map
        subgraph.inv_label_map = self.inv_label_map
        subgraph.depth_map = defaultdict(set)
        root = self.name_to_vertex[str(instance)]
        to_explore = { root }
        explored =  { root }
        subgraph.add_vertex( root )
        subgraph.root = root
        for d in range(depth):
            new_explore = set()
            for v in list(to_explore):
                subgraph.depth_map[d].add(v.get_name())
                for neighbor in self.get_neighbors(v):
                    subgraph.add_vertex(neighbor)
                    subgraph.add_edge(v, neighbor)
                    #if neighbor not in explored:
                    #    new_explore.add(neighbor)
                    #    explored.add(neighbor)
                    new_explore.add(neighbor)
            to_explore = new_explore
        
        return subgraph

    @staticmethod
    def rdflib_to_kg(rdflib_g, label_predicates=[]):
        # TODO: Make sure to filter out all tripels where p in label_predicates!
        # Iterate over triples, add s, p and o to graph and 2 edges (s-->p, p-->o)
        kg = KnowledgeGraph()
        for (s, p, o) in rdflib_g:
            if p not in label_predicates:
                if isinstance(s, rdflib.term.BNode):
                    s_v = Vertex(str(s), wildcard=True)
                elif isinstance(s, rdflib.term.Literal):
                    s_v = Vertex(str(s), literal=True)
                else:
                    s_v = Vertex(str(s))
                    
                if isinstance(o, rdflib.term.BNode):
                    o_v = Vertex(str(o), wildcard=True)
                elif isinstance(s, rdflib.term.Literal):
                    o_v = Vertex(str(o), literal=True)
                else:
                    o_v = Vertex(str(o))
                    
                p_v = Vertex(str(p), predicate=True, _from=s_v, _to=o_v)
                kg.add_vertex(s_v)
                kg.add_vertex(p_v)
                kg.add_vertex(o_v)
                kg.add_edge(s_v, p_v)
                kg.add_edge(p_v, o_v)
        return kg


############################################################################
#                      Hop, Walk and Tree objects                          #
############################################################################

class Hop:
    def __init__(self, vertex, wl_depth=0, root=False, wildcard=False):
        self.vertex = vertex
        self.wl_depth = wl_depth
        self.root = root
        self.wildcard = wildcard
        
    def to_str(self, kg):
        if self.wl_depth > 0:
            return kg.label_map[self.vertex][self.wl_depth]
        elif self.wildcard: 
            return '*'
        elif self.root: 
            return 'root'
        else: 
            return self.vertex.get_name()
        
    def __eq__(self, other):
        if self.root and other.root:
            return True
        
        return (self.vertex.name == other.vertex.name 
                and self.wl_depth == other.wl_depth 
                and self.wildcard == other.wildcard 
                and self.root == other.root)
            
    def __hash__(self):
        if self.root:
            return hash('root')
        if self.wildcard:
            return hash(('*', self.wl_depth))
        else:
            return hash((self.vertex.name, self.wl_depth))
            

class Walk(list):
    def __hash__(self):
        return hash(tuple(self))
    
    def to_str(self, kg):
        representation = []
        for hop in self:
            representation.append(hop.to_str(kg))
        return tuple(representation)
    
    
    def upper_ig(self, features, n_instances, labels, prior_entropy):
        for label in set(labels):
            if features[0].count(label) > features[1].count(label):
                features[0].extend([label] * Counter(labels)[label])
            else:
                features[1].extend([label] * Counter(labels)[label])

        pos_frac = len(features[1]) / n_instances
        pos_entr = entropy(np.unique(features[1], return_counts=True)[1])
        neg_frac = len(features[0]) / n_instances
        neg_entr = entropy(np.unique(features[0], return_counts=True)[1])
        return prior_entropy - (pos_frac * pos_entr + neg_frac * neg_entr)
    
    def calc_ig(self, kg, instances, labels, prior_entropy=None, 
                min_samples_split=0, current_best=None, cache=None):
        if prior_entropy is None:
            prior_entropy = entropy(np.unique(labels, return_counts=True)[1])
        
        features = {0: [], 1: []}
        for i, (inst, label) in enumerate(zip(instances, labels)):
            
            if cache is not None and (inst._id, len(self), self[-1].vertex.get_name()) in cache:
                found = cache[(inst._id, len(self), self[-1].vertex.get_name())]
            else:
                found = int(inst.find_walk(self))
                if cache is not None:
                    cache[(inst._id, len(self), self[-1].vertex.get_name())] = found

            features[found].append(label)

            # if current_best is not None and current_best > 0:
            #     ub = self.upper_ig(copy.deepcopy(features), len(instances), labels[i + 1:], prior_entropy)
            #     if ub < current_best:
            #         #print('prune!')
            #         return 0
            
        if len(features[1]) > min_samples_split and len(features[1]) < len(instances):
            pos_frac = len(features[1]) / len(instances)
            pos_entr = entropy(np.unique(features[1], return_counts=True)[1])
            neg_frac = len(features[0]) / len(instances)
            neg_entr = entropy(np.unique(features[0], return_counts=True)[1])
            ig = prior_entropy - (pos_frac * pos_entr + neg_frac * neg_entr)
            return ig
        else:
            return 0


class Tree():
    def __init__(self, walk=None, _class=None):
        self.left = None
        self.right = None
        self._class = _class
        self.walk = walk
        
    def evaluate(self, sample, kg):
        if self.walk is None:
            return self._class
        
        if sample.find_walk(self.walk):
            return self.right.evaluate(sample, kg)
        else:
            return self.left.evaluate(sample, kg)

############################################################################
#                                 Cache                                    #
############################################################################

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return None

    def set(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value