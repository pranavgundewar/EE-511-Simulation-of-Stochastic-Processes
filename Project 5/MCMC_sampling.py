"""
@author: Pranav Gundewar
Project #5: Optimization & Sampling via MCMC
Q1- MCMC for Sampling
"""

#%%
 #Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, cauchy, norm, logistic

#%%
def function(x):
   """
   Define Input mixture Distribution to generate samples
   unnormalized density function
   """
   return (0.6*beta.pdf(x, 1, 8) + 0.4*beta.pdf(x, 9, 1))

def proposal(x):
   """
   This is symmetric proposal PDF to plot the sample path
   """
#   return cauchy.rvs(loc=x, scale=0.15, size=1)
#   return norm.rvs(loc=x, scale=0.2, size=1)
   return logistic.rvs(loc=x, scale=1, size=1)

def metropolis(function, proposal, old):
    """
    basic metropolis algorithm, needs symmetric proposal distribution.
    """
    new = proposal(old)
    alpha = np.min([function(new)/function(old), 1])
    u = np.random.uniform()
    # _cnt_ indicates if new sample is used or not.
    cnt = 0
    if (u <= alpha):
        old = new
        cnt = 1
    return old, cnt

def run_chain(f, proposal, initial, n):
    """
    f:unnormalized density function to sample
    proposal:proposal distirbution
    start:initial start of the Markov Chain
    n:length of the chain
    """
    count = 0
    samples = [initial]
#    while (count != n-1):
##    for i in range(n):
#        initial, c = metropolis(f, proposal, initial)
#        count = count + c
#        if c==1:
#            samples.append(initial)
#    return samples
    for i in range(n):
        initial, c = metropolis(f, proposal, initial)
        count = count + c
        if i%1 is 0:
            samples.append(initial)
    return samples, count

#%%
#x = np.linspace(0, 1, 1000)
#plt.figure(1)
#plt.plot(x,function(x)) 
#plt.xlabel('x'); plt.ylabel('f(x)')
#plt.title('Input Mixture Distribution') 

#%%
while True:
   x0 = np.random.uniform()
   if function(x0) != 0:
      break
samples = run_chain(function, proposal, x0, 50000)

#%%
plt.figure(2)
plt.plot(samples[0],function(samples[0]),'o') 
plt.xlabel('x'); plt.ylabel('f(x)')
plt.title('Sample Path for Logistic Proposal PDF') 

#%%
plt.figure(3)
#plt.plot(samples[0],function(samples[0])) 
#plt.xlabel('x'); plt.ylabel('f(x)')
#plt.title('Sample Path for Logistic Proposal PDF')

