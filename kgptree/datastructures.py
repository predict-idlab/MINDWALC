import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp

from collections import defaultdict, Counter, OrderedDict
from functools import lru_cache
import heapq

import os
import itertools
import time

import rdflib

from scipy.stats import entropy

# The idea of using a hashing function is taken from:
# https://github.com/benedekrozemberczki/graph2vec
from hashlib import md5
import copy


class Vertex(object):
    
    def __init__(self, name, predicate=False, _from=None, _to=None):
        self.name = name
        self.predicate = predicate
        self._from = _from
        self._to = _to
        
    def __eq__(self, other):
        if other is None: 
            return False
        return self.__hash__() == other.__hash__()
    
    def get_name(self):
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

class Graph(object):
    _id = 0

    def __init__(self):
        self.vertices = set()
        self.transition_matrix = defaultdict(set)
        self.name_to_vertex = {}
        self.root = None
        self._id = Graph._id
        Graph._id += 1
        
    def add_vertex(self, vertex):
        if vertex not in self.vertices:
            self.vertices.add(vertex)            

        self.name_to_vertex[vertex.name] = vertex

    def add_edge(self, v1, v2):
        self.transition_matrix[v1].add(v2)

    def get_neighbors(self, vertex):
        return self.transition_matrix[vertex]

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

    def extract_neighborhood(self, instance, depth=8):
        neighborhood = Neighborhood()
        root = self.name_to_vertex[str(instance)]
        to_explore = { root }

        for d in range(depth):
            new_explore = set()
            for v in list(to_explore):
                if not v.predicate:
                    neighborhood.depth_map[d].add(v.get_name())
                for neighbor in self.get_neighbors(v):
                    new_explore.add(neighbor)
            to_explore = new_explore
        
        return neighborhood

    @staticmethod
    def rdflib_to_graph(rdflib_g, label_predicates=[]):
        kg = Graph()
        for (s, p, o) in rdflib_g:

            if p not in label_predicates:
                s = str(s)
                p = str(p)
                o = str(o)

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


class Neighborhood(object):
    def __init__(self):
        self.depth_map = defaultdict(set)
        
    def find_walk(self, vertex, depth):
        return vertex in self.depth_map[depth]


class Walk(object):
    def __init__(self, vertex, depth):
        self.vertex = vertex
        self.depth = depth

    def __eq__(self, other):
        return (hash(self.vertex) == hash(other.vertex) 
                and self.depth == other.depth)
    
    def __hash__(self):
        return hash((self.vertex, self.depth))

    def __lt__(self, other):
        return (self.depth, self.vertex) < (other.depth, other.vertex)


class TopQueue:
    def __init__(self, size):
        self.size = size
        self.data = []

    def add(self, x, priority):
        if len(self.data) == self.size:
            heapq.heappushpop(self.data, (priority, x))
        else:
            heapq.heappush(self.data, (priority, x))


class Tree():
    def __init__(self, walk=None, _class=None):
        self.left = None
        self.right = None
        self._class = _class
        self.walk = walk
        
    def evaluate(self, neighborhood):
        if self.walk is None:
            return self._class
        
        if neighborhood.find_walk(self.walk[0], self.walk[1]):
            return self.right.evaluate(neighborhood)
        else:
            return self.left.evaluate(neighborhood)

    @property
    def node_count(self):
        left_count, right_count = 0, 0
        if self.left is not None:
            left_count = self.left.node_count
        if self.right is not None:
            right_count = self.left.node_count
        return 1 + left_count + right_count