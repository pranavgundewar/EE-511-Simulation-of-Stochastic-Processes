# -*- coding: utf-8 -*-
"""
@author: Pranav Gundewar
Project 2
Q3- Double rejection
Convex summation of beta and triangle distribution
"""
# Importing Libraries
import numpy as np 
import random 
import math
import matplotlib.pyplot as plt

ar = np.zeros(1000)
at = 0 
rt = 0
def beta_pdf(x, a, b):
    """PDF of beta distribution."""
    return math.gamma(a+b)/(math.gamma(a)*math.gamma(b))*x**(a-1)*(1-x)**(b-1)

for i in range(100000):
    while (at < 1000):        
          a = 6*random.uniform(0,1)
          if (a > 0 and a <= 1):
              ar[i] = 0.5*beta_pdf(a,8,5)
          elif (a > 4 and a <= 5):
              ar[i] = 0.5*(a - 4)
          elif  (a > 5 and a <= 6):
              ar[i] = -0.5*(a - 6)
          else:
              ar[i] = 0
          if (ar[i] >= 1.46*random.uniform(0,1)):
                at = at + 1 
          else: 
                rt = rt + 1
      
plt.hist(ar, bins = 'auto')
k = rt / (rt + at) * 100
print('Rejection Rate: \n')
print(k)