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
        self.node_number = None
        
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
            right_count = self.right.node_count
        return 1 + left_count + right_count

    def visualize(self, output_path, _view=True, node_properties_function=None, meta_infos='', as_pdf=True):
        """
        Visualise the tree with [graphviz](http://www.graphviz.org/) and save it as pdf and or .gv file.

        **Params**
        ----------
          - `output_path` (string) - where the file needs to be saved
          - `_view` (boolean) - open the pdf after generation or not
          - `node_properties_function` (function) - function (params: xxx). Passed function returns a
            dict which contains the properties of the generated graph,
            using the Graphviz dot languade. see self._default_tree_visualization as example!
          - `meta_infos` (string) - a string, usually containing some usefully infos about the tree
            (which params used for decision tree classification training, which dataset used ...)
          - `as_pdf` (boolean) - if True, the output file will be a pdf % a .gv file, otherwise only one .gv file
        **Returns**
        -----------
            nothing
        """

        output_path = output_path + '.gv' if output_path.lower().endswith('.gv') else output_path

        if not node_properties_function:
            node_properties_function = self._default_node_visualization
        dot_code = self.convert_to_dot(node_properties_function, infos=meta_infos)
        if as_pdf:
            try:
                from graphviz import Source
                src = Source(dot_code)
                src.render(output_path, view=_view)
                return
            except Exception as e:
                print(f'An error occurred while trying to visualize a decision tree: \n{e}')
                print('Please install graphviz to use the datastructures.Tree.visualize method with as_pdf=True. ')

        with open(output_path, 'w') as f:
            f.write(dot_code)

        return

    def _default_node_visualization(self, node: Vertex):
        """
        Default function to visualize a decision tree node.
        uses the node ids for naming.
        :param node: the node to visualize
        :return: a dict containing the visualisation-properties of the node
        """

        is_leaf_node = True if node._class else False
        out = {'label': (node._class if is_leaf_node else node.walk[0] + f'\nd = {int(node.walk[1])}'),
               'fillcolor': '#DAE8FC' if is_leaf_node else '#FFF2CC',
               'color': '#6C8EBF' if is_leaf_node else "#D6B656",
               'style': "rounded,filled",
               'shape': 'ellipse' if is_leaf_node else 'box'}
        return out

    def convert_to_dot(self, label_func=None, font='Times-Roman', infos=''):
        """Converts a decision tree object to DOT code

        **Params**
        ----------
          - label_func (function) - function to label the nodes of the tree
        **Returns**
        -----------
            a string with the dot code for the decision tree
        """

        if not label_func:
            label_func = self._default_node_visualization

        self.nummerate_nodes_of_tree()
        s = 'digraph DT{\n'
        s += f'label="{infos}"\nfontname="{font}"\n'
        s += f'node[fontname="{font}"];\n'
        s += self._convert_node_to_dot(label_func)
        s += '}'
        return s

    def nummerate_nodes_of_tree(self, count=1):
        """
        Nummerate the nodes of the tree in order to give them unique names / ids.
        Updates the node_number attribute of each tree node.
        :param count:
        :return:
        """

        self.node_number = count
        if not self._class:
            self.left.nummerate_nodes_of_tree(count=count + 1)
            amount_subnodes_left = self.left.node_count
            self.right.nummerate_nodes_of_tree(count=count + amount_subnodes_left + 1)

    def _convert_node_to_dot(self, node_vis_props):
        """Convert node to dot format in order to visualize our tree using graphviz
        :param count: parameter used to give nodes unique names
        :return: intermediate string of the tree in dot format, without preamble (this is no correct dot format yet!)
        """

        num = self.node_number
        if self._class:  # leaf node:
            node_render_props = node_vis_props(self)
            node_props_dot = str([f'{k}="{node_render_props[k]}"' for k in node_render_props.keys()]).replace("'", '')
            s = f'Node{str(num)} ' + node_props_dot + ';\n'
        else:  # decision node:
            node_render_props = node_vis_props(self)
            node_props_dot = str([f'{k}="{node_render_props[k]}"' for k in node_render_props.keys()]).replace("'", '')
            s = f'Node{str(num)} ' + node_props_dot + ';\n'
            s += self.left._convert_node_to_dot(node_vis_props)
            s += 'Node' + str(num) + ' -> ' + 'Node' + str(num + 1) + ' [label="false"];\n'
            amount_subnodes_left = self.left.node_count
            s += self.right._convert_node_to_dot(node_vis_props)
            s += 'Node' + str(num) + ' -> ' + 'Node' + str(num + amount_subnodes_left + 1) + ' [label="true"];\n'

        return s
