# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:23:06 2023

@author: vishwas
"""
import scipy.linalg
import numpy as np
import networkx as nx

pi = np.zeros((4,4))

P0 = 0.5
P1 = 0.5
Ps1 = 0.05
Ph1 = 0.45
Pt1 = 0.5
m = 25

k = [2,3,5,9]
d = [5,10,15,20]




def calculate_pi(k, d):
# Create a directed graph
    G = nx.DiGraph()
    
    # Add edges with transition probabilities
    # numbering in the sequence otherwise it 
    # will not give desired result
    
    # Add nodes to the graph
   
    i = 0
    b = 1
    n = d*k
    p = m*k
    while b == 1:
        if i <= d*k:
            G.add_edge(i, i, weight=P0)
            G.add_edge(i, k+i, weight=P1)
            i = k+i
        elif n<p-k:
            n += 1
            G.add_edge(n, n-1, weight=Ps1)
            G.add_edge(n, n+k-1, weight=Ph1)
            G.add_edge(n, 0, weight=Pt1)
            
            
        else:
            # n += 1
            if n == p:
                # G.add_edge(n, n-1, weight=Ps1)
                # G.add_edge(n, n, weight=Ph1)
                # G.add_edge(n, 0, weight=Pt1)
                b=0
            else:
                n += 1
                G.add_edge(n, n-1, weight=Ps1)
                G.add_edge(n, p, weight=Ph1)
                G.add_edge(n, 0, weight=Pt1)
                
            # n += 1
            # k -= 1
            # if n == p:
            #     G.add_edge(n, n-1, weight=Ps1)
            #     G.add_edge(n, n, weight=Ph1)
            #     G.add_edge(n, 0, weight=Pt1)
            #     b=0
           
    # G.add_edge(i-1, i-2, weight=Ps1)
    # G.add_edge(i-1, i, weight=Ph1)
    # G.add_edge(i-1, 0, weight=Pt1)
    # G.add_edge(i, i-1, weight=Ps1)
    # G.add_edge(i, i, weight=Ph1)
    # G.add_edge(i , 0, weight=Pt1)           
       
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
    # print("Transition Matrix:")
    # print(transition_matrix)
    # idx = np.where(transition_matrix[:, 0]==0.5)[0]
    # idx = idx[1]
    
    values, left = scipy.linalg.eig(transition_matrix, right = False, left = True)
    
    # print("left eigen vectors = \n", left, "\n")
    # print("eigen values = \n", values)
    
    pi = left[:,0]
    pi_normalized = [(x/np.sum(pi)).real for x in pi]
    pi_d = sum(pi_normalized)
    return pi_d,transition_matrix,G
    # print("PI_i = ",pi_d)

for j in range(len(k)):
    for a in range(len(d)):
        pi[j,a],tm, G = calculate_pi(k[a],d[a])