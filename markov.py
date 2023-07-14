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

# Create a directed graph
G = nx.DiGraph()

# Add edges with transition probabilities
# numbering in the sequence otherwise it 
# will not give desired result
G.add_edge('state0', 'state0', weight=P0)
G.add_edge('state0', 'state2', weight=P1)
G.add_edge('state2', 'state2', weight=P0)
G.add_edge('state2', 'state4', weight=P1)
G.add_edge('state4', 'state4', weight=P0)
G.add_edge('state4', 'state6', weight=P1)
G.add_edge('state6', 'state6', weight=P0)
G.add_edge('state7', 'state6', weight=Ps1)
G.add_edge('state6', 'state8', weight=P1)
G.add_edge('state7', 'state9', weight=Ph1)
G.add_edge('state7', 'state0', weight=Pt1)
G.add_edge('state8', 'state7', weight=Ps1)
G.add_edge('state8', 'state10', weight=Ph1)
G.add_edge('state8', 'state0', weight=Pt1)
G.add_edge('state9', 'state8', weight=Ps1)
G.add_edge('state9', 'state10', weight=Ph1)
G.add_edge('state9', 'state0', weight=Pt1)
G.add_edge('state10', 'state9', weight=Ps1)
G.add_edge('state10', 'state10', weight=Ph1)
G.add_edge('state10', 'state0', weight=Pt1)
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
print(pi_normalized)
