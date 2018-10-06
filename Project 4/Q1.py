"""
@author: Pranav Gundewar
Project #4: Investigations on Monte Carlo Methods
Q1- Pi- Estimation
"""
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import random
from math import sqrt
############################################## PART A ##########################################

Pi_Array = np.zeros((50,1))
colors = ['g', 'b', 'y', 'c', 'm']
for i in range(0,50):
    cnt = 0 
    a = np.random.uniform(0,1,size=(100,2))
    label = np.ones(100)
    for j in range(0,100):
        if a[j][0] ** 2 + a[j][1] ** 2 <= 1:
            cnt +=1
            label[j] = 2            
    Pi_Array[i] = cnt*4/100
b = np.mean(Pi_Array)
plt.figure(1)
circle1 = plt.Circle((0, 0), 1, color='r',fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle1)
plt.legend([]['In', 'Out'])
plt.scatter(a[:,0],a[:,1], c = 'g')
for j in range(len(a)):
    if label[j] == 2:
        plt.scatter(a[j, 0], a[j, 1], c='r')

plt.figure(2)
circle1 = plt.Circle((0, 0), 1, color='r',fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle1)
plt.hist(Pi_Array)  
plt.title("Estimated Pi value = {:.4f}".format(b))
plt.xlabel("Pi Estimate Value")
plt.ylabel("Frequency")
plt.show()

print('\nEstimated Pi value = ',np.mean(Pi_Array))

############################################### PART B ##########################################
Pi_array = np.zeros((50,1))
N = list(range(100,10000,100))
Var_array = []
#Var_array = np.zeros((50,1))
for nums in N:
    for k in range(0,50):
        Pi = 0
        a = np.random.uniform(0,1,size=(nums,2))
        for i in range(0,nums):
            if a[i][0] ** 2 + a[i][1] ** 2 <= 1:
                Pi += 1
        Pi_array[k] = Pi/(nums/4.0)

    Est_var = 0
    mean = np.mean(Pi_array)
    for unit in Pi_array:
        Est_var += (unit - mean)**2
    Var_array.append(Est_var/49.0)

plt.plot(N,Var_array)
plt.title("Sample Variance of the Pi-estimates For Different Values of Samples")
plt.xlabel("Number of Samples")
plt.ylabel("Sample Variance")
plt.show()

