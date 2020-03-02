import networkx as nx


def graph_from_edgelist(path):
    
    """
    Build a graph starting from an edgelist which, for each row, have this format: <edge1><whitespace><edge2>\n, 
    a networkx graph is returned
    :param 
        path: the path to the edgelist
    :return: the built graph
    """

    G = nx.DiGraph()
    with open(path, 'r') as inp:
        lines = inp.readlines()
        lines = [l.replace('\n', '') for l in lines]
        for line in lines:
            nodes = line.split(' ')
            for node in nodes:
                if not node in G.nodes():
                    G.add_node(node)
            if not G.has_edge(nodes[0], nodes[1]):
                G.add_edge(nodes[0], nodes[1])
            else:
                print(nodes) 
    return G

def remove_void_types(G, void_types):

    """
    remove nodes which have no entities, 
    a tree is needed as return, so nodes are removed only if they are leaves or if all descendant are void
    :param 
        G: the tree to be pruned
        void_types: a list of nodes
    :return: the pruned tree
    """

    pruned_G = G.copy()
    for t in void_types:
        if len(list(pruned_G.successors(t))) == 0:
            pruned_G.remove_node(t)
        elif all(elem in void_types for elem in list(nx.algorithms.descendants(pruned_G,t))):
            pruned_G.remove_node(t)
    return pruned_G