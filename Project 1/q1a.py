# -*- coding: utf-8 -*-
"""
@author: Pranav Gundewar
@course; EE 511 Simulation Methods of Stochastic Systems
Project 1
Q1 Part A 
Simulation of fair Bernoulli trial  
"""

# Importing Libraries
import numpy as np 
import random
import matplotlib.pyplot as plt

# Initializing Variables
b = np.empty([100]);
h = 0
for i in range(100):
    a = random.uniform(0,1)                      # Generating random numbers between 0 to 1 with uniform districution
    if (a > 0.5):                            # Chechking for success
        h = h + 1
        b[i] = 1
    else:
        b[i] = 0
            
print("Number of heads: ",h)
print("Number of tails: ",100-h)

# Plotting the histogram 
plt.hist(b, bins = 'auto', facecolor='green')
plt.xlabel('Bernoulli Trials')
plt.ylabel('Count')
plt.title('Histogram of Bernoulli Trials: samples=100')
plt.grid(True)
plt.savefig('Q1a.jpeg')
plt.show()