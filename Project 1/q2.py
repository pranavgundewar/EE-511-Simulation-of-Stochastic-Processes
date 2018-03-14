# -*- coding: utf-8 -*-
"""
@author: Pranav Gundewar
Project 1
Q2- Counting Successes
Routine to find random number of edges selected 
"""
# Importing Libraries
import numpy as np 
import random
import matplotlib.pyplot as plt

# Initializing Variables
sum = 300                                               # Number of sum 
h = np.zeros([sum])
k = 50                                                   # Number of samples

# Checking sucess counting random variable
for i in range(sum):
    for j in range(k):
        a = random.random()                         # Generating random numbers between 0 to 1 with uniform districution
        if (a > 0.5):                                   # Chechking for success
            h[i] = h[i] + 1;
                
# Plotting the histogram 
plt.hist(h, bins = 'auto', facecolor='green')
plt.xlabel('Bernoulli sucess counting random variable')
plt.ylabel('Count')
plt.title('Histogram of sum = 300, k = 50')
plt.grid(True)
plt.savefig('Q2(k=50).jpeg')
plt.show()