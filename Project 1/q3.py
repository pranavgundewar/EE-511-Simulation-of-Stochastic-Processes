# -*- coding: utf-8 -*-
"""
@author: Pranav Gundewar
Project 1
Q3- Networking Part 1
Routine to find random number of edges selected 
"""
# Importing Libraries
import numpy as np 
import random
import matplotlib.pyplot as plt

# Initializing Variables
n = 20                                              # Number of people in group
N = (int) (n*(n-1)/2)
r = 100                                             # Number of Samples
print('Total number of possible edges: ',N)
h = np.zeros([r])                                   # Initializing an array for number of edges

# Checking number of edges for r samples of N trials
for i in range(r):
    for j in range(N):
        a = random.uniform(0,1)                     # Generating random numbers between 0 to 1 with uniform districution
        if (a < 0.05):                              # Chechking if p<0.05 to select an edge
                h[i] = h[i] + 1;
                
# Plotting the histogram 
plt.hist(h, bins = 'auto', facecolor='green')
plt.xlabel('Routine for random number of edges selected')
plt.ylabel('Count')
plt.title('Histogram for r =100 N =190')
plt.grid(True)
plt.savefig('Q3.jpeg')
plt.show()