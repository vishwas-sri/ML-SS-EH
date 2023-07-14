# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:23:06 2023

@author: vishwas
"""
import scipy.linalg
import numpy as np
import networkx as nx

P0 = 0.5
P1 = 0.5
Ps1 = 0.05
Ph1 = 0.45
Pt1 = 0.5
m = 5
k = 2
delta = 3
# Create a directed graph
G = nx.DiGraph()

# Add edges with transition probabilities
# numbering in the sequence otherwise it 
# will not give desired result
i = 0

while i<=m*k:
    if i <= delta*k:
        G.add_edge(i, i, weight=P0)
        G.add_edge(i, k+i, weight=P1)
        i = k+i
    else:
        G.add_edge(i-1, i-2, weight=Ps1)
        G.add_edge(i-1, i+1, weight=Ph1)
        G.add_edge(i-1, 0, weight=Pt1)
        i += 1
        if 
    else:
        G.add_edge(i-1, i-2, weight=Ps1)
        G.add_edge(i-1, i, weight=Ph1)
        G.add_edge(i-1, 0, weight=Pt1)
    #     G.add_edge(i, i-1, weight=Ps1)
    #     G.add_edge(i, i, weight=Ph1)
    #     G.add_edge(i, 0, weight=Pt1)
        
    # G.add_edge(2, 2, weight=P0)
    # G.add_edge(2, 4, weight=P1)
# G.add_edge('state4', 'state4', weight=P0)
# G.add_edge('state4', 'state6', weight=P1)
# G.add_edge('state6', 'state6', weight=P0)
# G.add_edge('state7', 'state6', weight=Ps1)
# G.add_edge('state6', 'state8', weight=P1)
# G.add_edge('state7', 'state9', weight=Ph1)
# G.add_edge('state7', 'state0', weight=Pt1)
# G.add_edge('state8', 'state7', weight=Ps1)
# G.add_edge('state8', 'state10', weight=Ph1)
# G.add_edge('state8', 'state0', weight=Pt1)
# G.add_edge('state9', 'state8', weight=Ps1)
# G.add_edge('state9', 'state10', weight=Ph1)
# G.add_edge('state9', 'state0', weight=Pt1)
# G.add_edge('state10', 'state9', weight=Ps1)
# G.add_edge('state10', 'state10', weight=Ph1)
# G.add_edge('state10', 'state0', weight=Pt1)
# ... continue adding edges for other states and their transition probabilities

# Get the list of states
states = list(G.nodes)

# Create the transition matrix
transition_matrix = np.zeros((len(states), len(states)))

# Fill in the transition matrix using the edge weights
for i, state in enumerate(states):
    successors = G[state]
    total_weight = sum(successors[next_state]['weight'] for next_state in successors)
    for j, next_state in enumerate(states):
        if next_state in successors:
            transition_matrix[i, j] = successors[next_state]['weight'] / total_weight

# Check if the transition probabilities are valid
if not np.allclose(np.sum(transition_matrix, axis=1), 1):
    print("Invalid transition probabilities. The probabilities should sum to 1.")
else:
    print("Valid transition probabilities.")

# Display the transition matrix
print("Transition Matrix:")
print(transition_matrix)


values, left = scipy.linalg.eig(transition_matrix, right = False, left = True)

# print("left eigen vectors = \n", left, "\n")
# print("eigen values = \n", values)

pi = left[:,0]
pi_normalized = [(x/np.sum(pi)).real for x in pi]
pi_d = sum(pi_normalized[4:])
print(pi_d)
