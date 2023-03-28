__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

import biofuzznet.biofuzznet as biofuzznet
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple

# %% global constants
DEFAULT_EDGE_COLOR_SCHEME = {
    "simple": "k",
    "transfer_function": "grey",
    "positive": "k",
    "negative": "red",
}
DEFAULT_NODE_SHAPE_SCHEME = {
    "biological": "s",
    "logic_gate_AND": "o",
    "logic_gate_OR": "d",
    "logic_gate_NOT": "^",
}
DEFAULT_NODE_COLOR_SCHEME = {
    "biological": "lightsteelblue",
    "logic_gate_AND": "lightgrey",
    "logic_gate_OR": "mistyrose",
    "logic_gate_NOT": "lightsalmon",
}


# %%
def assign_node_levels(G: biofuzznet.BioFuzzNet):
    G = G.copy()
    for n in G.nodes:
        G.nodes[n]["level"] = -1
    for root in G.root_nodes:
        G.nodes[root]["level"] = 0
        for n in G.nodes:
            try:  # Try statement to avoid problems related to n not being related to the current root node
                G.nodes[n]["level"] = 2 * max(
                    G.nodes[n]["level"],
                    nx.shortest_path_length(G, source=root, target=n),
                )
                if G.nodes[n]["node_type"] != "biological":
                    G.nodes[n]["level"] += 1
            except:
                continue
    # for leaf in G.leaf_nodes:
    #   G.nodes[leaf]["level"] = M + 1
    return G


def biological_graph_only(G: biofuzznet.BioFuzzNet):
    biol_G = biofuzznet.BioFuzzNet()
    for node in G.biological_nodes:
        biol_G.add_fuzzy_node(node, "BIO")
    old_edges = [e for e in G.edges]
    while old_edges:
        edge = old_edges.pop()
        # First case: transfer edge linking 2 biological nodes
        if (
            G.nodes[edge[0]]["node_type"] == "biological"
            and G.nodes[edge[1]]["node_type"] == "biological"
        ):
            biol_G.add_edge(edge[0], edge[1], edge_type="positive")
        elif (
            G.nodes[edge[0]]["node_type"] == "biological"
            and G.nodes[edge[1]]["node_type"] != "biological"
        ):
            biol_child = None
            found_biol = False
            curr_node = edge[1]
            while not found_biol:
                children = [s for s in G.successors(curr_node)]
                assert (
                    len(children) == 1
                )  # Should only be one by definition of logical node in a BFZ which was properly defined from a biological network
                # This might not always be respected for example BFZ used to test features
                child = children[0]
                found_biol = G.nodes[child]["node_type"] == "biological"
                if found_biol:
                    biol_child = child
                else:
                    curr_node = child
            if G.nodes[edge[1]]["node_type"] == "logic_gate_NOT":
                biol_G.add_edge(
                    edge[0], biol_child, edge_type="negative"
                )  # This is not a BFZ edge type. But only used in the draing function. Extend the class if I use this in other more directly exposed functions
            else:
                biol_G.add_edge(edge[0], biol_child, edge_type="positive")
        else:  # Then the first node is not biological; so there'll be a parent edge that will relate us back to the previous case
            continue
    return biol_G


def draw_BioFuzzNet(
    G: biofuzznet.BioFuzzNet,
    edge_color_scheme: dict = DEFAULT_EDGE_COLOR_SCHEME,
    node_shape_scheme: dict = DEFAULT_NODE_SHAPE_SCHEME,
    node_color_scheme: dict = DEFAULT_NODE_COLOR_SCHEME,
    figsize: Tuple = (15, 15),
    biological_only: bool = False,
    connection_style="arc3, rad = 0.25",
):
    # Cannot constrain G to have a BioFuzzNet class, otherwise there will be a circular import
    """
    Draws the BioFuzzNet.

    Args:
       edge_color_scheme: a dict associating the 'edge_type' attribute of BioFuzzNet edges to a color
       node_shape_scheme: a dict associating the 'node_type' attribute of BioFuzzNet nodes to a shape
       figsize: a Tuple containing the size of the desired figure
       biological_only: boolean indicating whether only the biological nodes should be drawn, in which case the edges are either negative or positive. Information about the AND and OR gates is however lost.
       connection_style: style of the FancyArrow used to draw the edges. see the documentation for networkx function draw_networkx_edges
    Returns:
        dictionnary of node positions keyed by nodes
    """
    plt.figure(figsize=figsize)
    nodesize = 800

    # Set up the layout
    if not biological_only:
        G = assign_node_levels(G)  # Returns a different graph
    else:
        G = biological_graph_only(G)
        G = assign_node_levels(G)

    pos = nx.multipartite_layout(G, subset_key="level", align="horizontal", scale=2)
    node_type_list = list(node_shape_scheme.keys())

    # Draw the nodes
    for node_type in node_type_list:
        nodes_to_plot = [
            node
            for node, attributes in G.nodes(data=True)
            if attributes["node_type"] == node_type
        ]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes_to_plot,
            node_shape=node_shape_scheme[node_type],
            node_color=node_color_scheme[node_type],
            edgecolors="black",
            node_size=nodesize,
        )
    # Draw the edges and the labels
    edge_colors = [edge_color_scheme[G[u][v]["edge_type"]] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        arrowsize=15,
        node_size=nodesize,
        min_source_margin=0,
        min_target_margin=0,
        connectionstyle=connection_style,
    )
    nx.draw_networkx_labels(G, pos, font_size=7)
    return pos
