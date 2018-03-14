# -*- coding: utf-8 -*-
"""
@author: Pranav Gundewar
Project 2
Q2- Waiting time exponential distribution
Routine to Chi-Square GOF and distribution of count 
"""
# Importing Libraries
import numpy as np 
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.stats import expon

# Initializing Variables and definitions

def expon_pdf(x, lmabd=5):
    """PDF of exponential distribution."""
    return lmabd*np.exp(-lmabd*x)

def expon_cdf(x, lambd=5):
    """CDF of exponetial distribution."""
    return 1 - np.exp(-lambd*x)

def expon_icdf(u, lambd=5):
    """Inverse CDF of exponential distribution - i.e. quantile function."""
    return -np.log(1-u)/lambd

# Finding Inverse CDF and random generated PDF
u = np.random.random(1000)
v = np.zeros(1000)
c = np.zeros(1000)
for i in range (1000):
    u = rand.uniform(0,1)
    v[i] = expon_icdf(u)

r = expon.rvs(size=1000,scale = 0.2)

# Finding Count
for i in range (1000):
    b = 0
    cntr = 0
    while (b <= 1):
        u = rand.uniform(0,1)
        a = expon_icdf(u)
        b = b + a
        cntr = cntr + 1
    c[i] = cntr
plt.figure(figsize = (12,4))
plt.hist(c, bins = 'auto', linewidth=2, facecolor='green')
plt.xlabel('Arrival Time Count')
plt.ylabel('Occurance')
plt.title('Count of Exponentially distributed time interval of 1 unit time');

# Plotting Histogram

plt.figure(figsize = (12,4))
plt.subplot(121)
plt.title('Histogram of exponential PDF using inverse method');
binsn = [0.04, 0.1, 0.18, 0.32]
#(x1, y1, z1) = plt.hist(v, bins = 'sturges', linewidth=2, facecolor='green')
(x1, y1, z1) = plt.hist(v, binsn, linewidth=2, facecolor='green')
plt.subplot(122)
plt.title('Histogram of exponential PRNGs');
#(x2, y2, z2) = plt.hist(r, bins = 'sturges', linewidth=2, facecolor='blue')
(x2, y2, z2) = plt.hist(r, binsn, linewidth=2, facecolor='blue')
l = chisquare(x1,x2)
print (l)
#Power_divergenceResult(statistic=10.290621774943048, pvalue=0.41537674668092484)
#[ 427.  224.  152.   81.   44.   34.   19.    4.    7.    6.    2.]
#[ 416.  249.  141.   82.   45.   25.   21.    8.    6.    6.    1.]

