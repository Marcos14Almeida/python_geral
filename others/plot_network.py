# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:16:23 2022

@author: marcos

Plot different networks configurations in the form of graphs
"""
# =============================================================================
# ================================= Libraries =================================
# =============================================================================

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
#                                     Main
# =============================================================================
# %%
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (1, 3), (1, 4)])
pos = {1: (20, 30), 2: (40, 30), 3: (30, 10), 4: (0, 40)}

nx.draw_networkx(G, pos=pos)

plt.show()
plt.close()

# %%
G = nx.Graph()
G.add_edge(0, 1, color='r', weight=5)
G.add_edge(1, 2, color='k', weight=2)
G.add_edge(2, 3, color='k', weight=2)
G.add_edge(3, 4, color='k', weight=2)
G.add_edge(4, 0, color='r', weight=5)

colors = nx.get_edge_attributes(G, 'color').values()
weights = nx.get_edge_attributes(G, 'weight').values()

pos = nx.circular_layout(G)
nx.draw(G, pos,
        edge_color=colors,
        width=list(weights),
        with_labels=True,
        node_color='lightgreen')
plt.show()
plt.close()

# %%
# https://stackoverflow.com/questions/13974643/networkx-draw-text-on-edges

# Sample graph
G = nx.Graph()
G.add_edge(0, 1)
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(1, 3)

labels = {(0, 1): '-28.3', (2, 3): '-35.2'}

pos = nx.spring_layout(G)

nx.draw(G, pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=16)

plt.title("Service 12")
plt.show()
plt.close()

# %%
# Number of node connections
plt.hist([v for k, v in nx.degree(G)])
nx.diameter(G)  # Farthest distance
plt.show()
plt.close()

# %%


nodes = list(range(50))
df = pd.DataFrame({
    'from': np.random.choice(nodes, 50),
    'to': np.random.choice(nodes, 50)
    })

G = nx.from_pandas_edgelist(df, source='from', target='to')

nx.draw(G)
plt.show()
plt.close()
