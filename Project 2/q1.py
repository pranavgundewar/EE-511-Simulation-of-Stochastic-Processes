# -*- coding: utf-8 -*-
"""
@author: Pranav Gundewar
Project 2
Q1- Networking Part 2
Routine to plot network graph and histogram of degree vertex
"""
# Importing Libraries
import numpy as np 
import random as rand
import matplotlib.pyplot as plt
import networkx as nx

# Initializing Variables
n = 50                                  # Nodes
N = (int) (n*(n-1)/2) 
G1 = nx.Graph()                         # initialize the graphs
G2 = nx.Graph()
G3 = nx.Graph()
d = np.zeros([n])                       # Array for 
#H = nx.path_graph(n)
#G1.add_nodes_from(H)
for i in range(n-1):
    for j in range(i+1,n):
        a = rand.uniform(0,1) ;
        if (a < 0.02):
            G1.add_edge(i,j)            # Add edge if it exits
        else :
            G1.add_node(i)
            G1.add_node(j)
        if (a < 0.09):
            G2.add_edge(i,j)
        else :
            G2.add_node(i)
            G2.add_node(j)
        if (a < 0.12):
            G3.add_edge(i,j)
        else :
            G3.add_node(i)
            G3.add_node(j)
        
options = { 'node_color': 'red','node_size': 100,'width': 3,}

nx.draw_circular(G1, with_labels=True, **options)
plt.title('Network Graph for n = 50, p = 0.02')
plt.show()

nx.draw_circular(G2, with_labels=True, **options)
plt.title('Network Graph for n = 50, p = 0.09')
plt.show()


nx.draw_circular(G3, with_labels=True, **options)
plt.title('Network Graph for n = 50, p = 0.12')
plt.show()

for i in range(n):
    d[i] = G1.degree[i]                     # Calculate Degree of Vertex

plt.hist(d, bins = 'auto', facecolor='green')   # Plotting histogram
plt.xlabel('Vertex Degree')
plt.ylabel('Number of Connections')
plt.title('n = 50, p = 0.12')
plt.grid(True)
plt.legend(n = 50, p = 0.02)
#plt.savefig('Q3.jpeg')
plt.show()


n = 100 
N = (int) (n*(n-1)/2) 
G4 = nx.Graph()
d = np.zeros([n])  
#H = nx.path_graph(n)
#G1.add_nodes_from(H)
for i in range(n-1):
    for j in range(i+1,n):
        a = rand.uniform(0,1) ;
        if (a < 0.06):
            G4.add_edge(i,j)
        else :
            G4.add_node(i)
            G4.add_node(j)

options = { 'node_color': 'red','node_size': 100,'width': 3,}
nx.draw_circular(G4, with_labels=True, **options)
plt.title('Network Graph for n = 100, p = 0.06')
plt.show()

for i in range(n):
    d[i] = G4.degree[i]

plt.hist(d, bins = 'auto', facecolor='green')
plt.xlabel('Vertex Degree')
plt.ylabel('Number of Connections')
plt.title('n = 100, p = 0.06')
plt.grid(True)
plt.legend(n = 100, p = 0.06)
#plt.savefig('Q3.jpeg')
plt.show()

