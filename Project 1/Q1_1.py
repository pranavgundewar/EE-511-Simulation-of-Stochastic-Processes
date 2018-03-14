# -*- coding: utf-8 -*-
"""
@author: Pranav Gundewar
@course; EE 511 Simulation Methods of Stochastic Systems
Project 1
Q1 Part B 
Routine to count number of successes  
"""

# Importing Libraries
import numpy as np 
import random
import matplotlib.pyplot as plt

# Initializing Variables

h = np.zeros([100])
for i in range(100):
    for j in range(7):
        a = random.uniform(0,1)                      # Generating random numbers between 0 to 1 with uniform districution
        if (a > 0.5):                                # Chechking for success
            h[i] = h[i]+1
            
# Plotting the histogram 
plt.hist(h, bins = 'auto', facecolor='green')
plt.xlabel('Number of Successes')
plt.ylabel('Count')
plt.title('Histogram of number of successe: samples=100')
plt.grid(True)
plt.savefig('Q1b.jpeg')
plt.show()
