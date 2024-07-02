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
    
    def __init__(self, name, predicate=False, _from=None, _to=None, relation_modified=False):
        self.name = name
        self.predicate = predicate
        self.relation_modified = relation_modified
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
    def rdflib_to_graph(rdflib_g, label_predicates=[], relation_tail_merging=True):
        '''
        Converts an rdflib graph to a Graph object.
        During the conversion, a multi-relation graph (head)-[relation]->(tail) (aka subject, predicate, object)is converted to a non-relational graph.
        e.g. converting it to (head)-->(relation)-->(tail), or, if apply_relation_tail_merging is True, to (head)-->(relation_tail).

        :param rdflib_g: An rdflib graph, e.g. loaded with rdflib.Graph().parse('file.n3')
        :param label_predicates: a list of predicates that are used as labels, and should not be converted to edges?
        :param relation_tail_merging: If true, relation-tail-merging is applioed, as described in the paper
        "Investigating and Optimizing MINDWALC Node Classification to Extract Interpretable DTs from KGs":
        The process of relation-tail merging works as follows: First, a specific tail node is
        selected, t, as well as a set of nr relations of identical type, r, where the topological
        form (*)-r->(t) is given. The process of relation-tail merging then involves inserting
        a new node, rt, so that (*)-r->(t) turns into (*)-->(rt)-->(t). The new directional
        edges, -->, are now typeless, and the new inserted node, rt, represents a relationmodified node and is
        named accordingly in the form <type_of_r>_<name_of_t>.
        :return: A Graph object of type datastructures::Graph
        '''

        kg = Graph()

        for (s, p, o) in rdflib_g:
            if p not in label_predicates:

                # Literals are attribute values in RDF, for instance, a person’s name, the date of birth, height, etc.
                if isinstance(s, rdflib.term.Literal) and not str(s):
                    s = "EmptyLiteral"
                if isinstance(p, rdflib.term.Literal) and not str(p):
                    p = "EmptyLiteral"
                if isinstance(o, rdflib.term.Literal) and not str(o):
                    o = "EmptyLiteral"

                s = str(s)
                p = str(p)
                o = str(o)

                s_v = Vertex(s)

                if relation_tail_merging:
                    o_v_relation_mod = Vertex(f"{p}_MODIFIED_{o}", relation_modified=True)
                    o_v = Vertex(o)
                    kg.add_vertex(s_v)
                    kg.add_vertex(o_v_relation_mod)
                    kg.add_vertex(o_v)
                    kg.add_edge(s_v, o_v_relation_mod)
                    kg.add_edge(o_v_relation_mod, o_v)
                else:
                    o_v = Vertex(o)
                    p_v = Vertex(p, predicate=True, _from=s_v, _to=o_v)
                    kg.add_vertex(s_v)
                    kg.add_vertex(p_v)
                    kg.add_vertex(o_v)
                    kg.add_edge(s_v, p_v)
                    kg.add_edge(p_v, o_v)
        return kg

    def graph_to_neo4j(self, uri='bolt://localhost', user='neo4j', password='password'):
        '''
        Converts the graph to a neo4j database. Needs an empty running neo4j db.
        :param uri: address where neo4j db is running
        :param user: username of neo4j db
        :param password: password of neo4j db
        :return: None
        '''

        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("Please install the neo4j-driver package to use this function.")
        from tqdm import tqdm

        use_nodes_for_predicates = True # if false, the predicates are used as edges. Otherwise as nodes.
        relation_name = 'R'

        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # check if db is empty:
            node_count = session.run("MATCH (n) return count(n)").single().value()
            if node_count > 0:
                print("Neo4j database is not empty, aborting graph to neo4h db convertion to avoid data loss.")
                return

            for v in self.vertices:
                if not v.predicate:
                    # name = v.name.split('/')[-1]
                    name = v.name.replace("'", "")
                    session.run(f"CREATE (a:Node" + (":RelationModified" if v.relation_modified else "") +
                                " {name: '" + name + "'})")  # .split(' ')[0] + '_' + vertex.__hash__()

            for v in tqdm(self.vertices):
                if not v.predicate:
                    # v_name = v.name.split('/')[-1]
                    v_name = v.name.replace("'", "")

                    node_type = "Node" + (":RelationModified" if v.relation_modified else "")

                    ids_v = [r["id(v)"] for r in
                             session.run(
                                 "MATCH (v:" + node_type + " {name: '" + v_name + "'}) where not (v:Predicate) RETURN id(v)")]
                    if len(ids_v) == 0:
                        raise Exception(f"no id found for {v_name}")
                    elif len(ids_v) == 1:
                        id_v = ids_v[0]
                    else:
                        raise Exception(f"multiple ids found for {v_name}: {ids_v}")

                    for pred in self.get_neighbors(v):

                        if pred.predicate:
                            pred_name = "".join(
                                [c for c in pred.name.split('/')[-1].replace("#", "_").replace('-', '_') if
                                 not c.isdigit()])
                            pred_name = pred_name[1:] if pred_name[0] in ["_", "-"] else pred_name

                            for obj in self.get_neighbors(pred):
                                # obj_name = obj.name.split('/')[-1]
                                obj_name = obj.name.replace("'", "")

                                ids_obj = [r["id(obj)"] for r in
                                           session.run(
                                               "MATCH (obj:Node {name: '" + obj_name + "'}) where not (obj:Predicate) RETURN id(obj)")]
                                if len(ids_obj) == 0:
                                    raise Exception(f"no id found for {obj_name}")
                                elif len(ids_obj) == 1:
                                    id_obj = ids_obj[0]
                                else:
                                    raise Exception(f"multiple ids found for {obj_name}: {ids_obj}")

                                if use_nodes_for_predicates:
                                    q = (f"MATCH (a), (b) WHERE ID(a)={id_v} AND ID(b)={id_obj} "
                                         "MERGE (a)-[:") + relation_name + "]->(c:Predicate {name: '" + pred_name + "'})-[:" + relation_name + "]->(b)"
                                else:
                                    q = f"MATCH (a), (b) WHERE ID(a)={id_v} AND ID(b)={id_obj} MERGE (a)-[:" + pred_name + "]->(b)"
                                session.run(q)

                        else:
                            obj_name = pred.name.replace("'", "")

                            ids_obj = [r["id(obj)"] for r in
                                       session.run(
                                           "MATCH (obj:Node {name: '" + obj_name + "'}) RETURN id(obj)")]
                            if len(ids_obj) == 0:
                                raise Exception(f"no id found for {obj_name}")
                            elif len(ids_obj) == 1:
                                id_obj = ids_obj[0]
                            else:
                                raise Exception(f"multiple ids found for {obj_name}: {ids_obj}")

                            q = f"MATCH (a), (b) WHERE ID(a)={id_v} AND ID(b)={id_obj} MERGE (a)-[:" + relation_name + "]->(b)"
                            session.run(q)

        driver.close()

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

if __name__ == "__main__":
    from tree_builder import MINDWALCTree, MINDWALCForest, MINDWALCTransform
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix
    import sys

    # load graph:
    rdf_file = 'data/AIFB/aifb.n3'
    _format = 'n3'
    label_predicates = [ # these predicates will be deleted, otherwise clf task would get to easy?
        rdflib.URIRef('http://swrc.ontoware.org/ontology#affiliation'),
        rdflib.URIRef('http://swrc.ontoware.org/ontology#employs'),
        rdflib.URIRef('http://swrc.ontoware.org/ontology#carriedOutBy')
    ]
    g = rdflib.Graph()
    g.parse(rdf_file, format=_format)

    # load train data:
    train_file = 'data/AIFB/AIFB_test.tsv'
    test_file = 'data/AIFB/AIFB_train.tsv'
    entity_col = 'person'
    label_col = 'label_affiliation'
    test_data = pd.read_csv(train_file, sep='\t')
    train_data = pd.read_csv(test_file, sep='\t')

    train_entities = [rdflib.URIRef(x) for x in train_data[entity_col]]
    train_labels = train_data[label_col]

    test_entities = [rdflib.URIRef(x) for x in test_data[entity_col]]
    test_labels = test_data[label_col]


    # convert to non relational graphs using relation-to-node convertion:
    kg = Graph.rdflib_to_graph(g, label_predicates=label_predicates, relation_tail_merging=False)
    #kg.graph_to_neo4j(password=sys.argv[1])
    verts_a = len(kg.vertices)
    print(f"generated graph using relation-to-node-convertion has {str(float(verts_a)/1000).replace('.', ',')} vertices")
    clf = MINDWALCTree(path_max_depth=6, min_samples_leaf=1, max_tree_depth=None, n_jobs=1)
    clf.fit(kg, train_entities, train_labels)
    preds = clf.predict(kg, test_entities)
    print(f"accuracy: {accuracy_score(test_labels, preds)}")

    print()

    # convert to non relational graphs using relation-tail-merging:
    kg = Graph.rdflib_to_graph(g, label_predicates=label_predicates, relation_tail_merging=True)
    verts_b = len(kg.vertices)
    print(
        f"generated graph using relation_tail_merging has {str(float(verts_b)/1000).replace('.', ',')} vertices")
    clf = MINDWALCTree(path_max_depth=6, min_samples_leaf=1, max_tree_depth=None, n_jobs=1)
    clf.fit(kg, train_entities, train_labels)
    preds = clf.predict(kg, test_entities)
    print(f"accuracy: {accuracy_score(test_labels, preds)}")

    print(f"\nrelation_tail_merging reduced the number of vertices by {verts_a - verts_b} ({round((verts_a - verts_b)/verts_a *100, 0)} %)")



