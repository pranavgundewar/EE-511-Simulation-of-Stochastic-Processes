# -*- coding: utf-8 -*-
"""
@author: Pranav Gundewar
Project 1
Q1 Part C 
Routine to count longest run of heads 
"""

# Importing Libraries
import numpy as np 
import random
import matplotlib.pyplot as plt

# Initializing Variables

lh = np.zeros([100])                                # Longest run of heads
h = np.zeros([100])

# Checking to count longest run of heads
for i in range(100):
    for j in range(7):
        a = random.uniform(0,1)                     # Generating random numbers between 0 to 1 with uniform districution
        if (a > 0.5):                               # Chechking for success
            h[i] = h[i]+1
        else:
            if (lh[i]<h[i]):
                lh[i] = h[i] 
            h[i] = 0

# Plotting the histogram                 
plt.hist(lh, bins = 'auto', facecolor='green')
plt.xlabel('Longest run of heads')
plt.ylabel('Count')
plt.title('Histogram of longest run of heads: samples=100')
plt.grid(True)
plt.savefig('Q1c.jpeg')
plt.show()

