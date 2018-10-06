"""
@author: Pranav Gundewar
Project #4: Investigations on Monte Carlo Methods
Q2- Monte Carlo Estimation and Varinace Reduction Stratergies
"""
# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import truncnorm
import statistics
from scipy.stats import multivariate_normal as mvn

###################################### PART A #################################
print('\nDifferent Monte Carlo Estimation for function 1/(1+sinh(2*x)*log(x)')
# Define function
def integrand1(x):
    return 1/(1+np.sinh(2*x)*np.log(x))

# N draws 
N= 1000
# Define limites for integrals
a1 = 0.8;    
b1 = 3; 

# Plot the function to get better idea about its estimation
x=np.linspace(a1,b1,1000)
plt.plot(x,integrand1(x))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function Graph: 1/(1+sinh(2*x)*log(x)')
plt.grid()
plt.show()
############################### Uniform Sampling ###########################
def estimate1():
    x = np.random.uniform(low=a1, high=b1, size=N) # N values uniformly drawn from a to b 
    z =integrand1(x)   # CALCULATE THE f(x) 
    V = b1-a1
    # Monte-Carlo Estimation
    I = V * np.sum(z) / N;
    return I

exactval=0.609553   # f(x) value calculated using Mathematica
I = np.empty(50)
for i in range(50):        
    I[i] = estimate1()
var = statistics.variance(I)
print("\nMonte Carlo estimation using Uniform Sampling")
print("Estimation = {:.6f}".format(np.mean(I)), "Variance = {:.6f}".format(var))

############################### Stratified Sampling ###########################

def strfsample1(sv,n):
    # Take more number of samples where function value is varying the most
    x1 = np.random.uniform(low=a1, high=sv, size=n)
    # Calculate the integral value for that section where value is varying the most
    V = sv - a1
    z = integrand1(x1)
    I1 = V * np.sum(z) / n 
    # Take rest of the samples from section of function where value is not changing by much
    x2 = np.random.uniform(low=sv, high=b1, size=N-n)
    V = b1 - sv
    z = integrand1(x2)
    I2 = V * np.sum(z) / (N-n) 
    # Monte-Carlo Estimation
    I = I1+ I2
    return I
I = np.empty(50)
for i in range(50):        
    I[i] = strfsample1(1.5,800);
var = statistics.variance(I)
print("Monte Carlo estimation using Stratified Sampling")
print("Estimation = {:.6f}".format(np.mean(I)),"Variance = {:.6f}".format(var))


############################### Importance Sampling ###########################


def impsample1():
    x = truncnorm.rvs(a1, b1, loc=0, size=N)
    p = truncnorm.pdf(x, a1, b1, loc=0)
    z = np.empty(N)
    for i in range(N):
        z[i] = integrand1(x[i])
        z[i] /= p[i]
    return np.mean(z)

I = np.empty(50)
for i in range(50):        
    I[i] = impsample1()
var = statistics.variance(I)
print("Monte Carlo estimation using Importance Sampling")
print("Estimation = {:.6f}".format(np.mean(I)),"Variance = {:.6f}".format(var))

###################################### PART B #################################
#Define function
def integrand2(x,y):
    return np.exp(-x**4-y**4)

#Define limites for integrals
a2 = -1*np.pi
b2 = np.pi
# use N draws 
N= 1000
exactval=3.28626
# Plot the function to get better idea about its estimation
x=np.linspace(a2,b2,1000)
y=np.linspace(a2,b2,1000)
plt.plot(x,integrand2(x,y))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function Graph: exp(-x^4-y^4)')
plt.grid()
plt.show()

print('\nDifferent Monte Carlo Estimation for function e^(-x^4-y^4)\n')

################################ Uniform Sampling ###########################

def estimate2():
    x = np.random.uniform(low=a2, high=b2, size=N) # N values uniformly drawn from a to b 
    y = np.random.uniform(low=a2, high=b2, size=N) # N values uniformly drawn from a to b 
    z =integrand2(x,y)   # CALCULATE THE f(x) 
    V = b2-a2
    I = V * V * np.sum(z)/ N;
    return I

I = np.empty(50)
for i in range(50):        
    I[i] = estimate2()
var = statistics.variance(I)
print("Monte Carlo estimation using Uniform Sampling")
print("Estimation = {:.6f}".format(np.mean(I)),"Variance = {:.6f}".format(var))

################################# Stratified Sampling ###########################
def strfsample2():      
    x1 = np.random.uniform(low=-1.1, high=1.1, size=1000)
    y1 = np.random.uniform(low=-1.1, high=1.1, size=1000)
    V = 2.2
    z = integrand2(x1,y1)
    I1 = V * V * np.sum(z) / 1000 
    return I1

I = np.empty(50)
for i in range(50):        
    I[i] = estimate2()
var = statistics.variance(I)
print("Monte Carlo estimation using Stratified Sampling")
print("Estimation = {:.6f}".format(np.mean(I)),"Variance = {:.6f}".format(var))
#
################################# Importance Sampling #########################
#
def impsample2():
    count = 0 
    X = np.empty((N,2))
    while count < N:
        x = mvn.rvs(mean = [0, 0] ,cov = [[1, 0], [0, 1]])
        if x[0] > a2 and x[0] < b2 and x[1] > a2 and x[1] < b2:
            X[count,0] = x[0]
            X[count,1] = x[1]
        count +=1
    p = mvn.pdf(X,mean = [0, 0] ,cov = [[1, 0], [0, 1]])
    I = integrand2(X[:,0],X[:,1])
    q = np.divide(I,p) 
    return np.mean(q)
    
I = np.empty(50)
for i in range(50):        
    I[i] = impsample2()
var = statistics.variance(I)
print("Monte Carlo estimation using Importance Sampling")
print("Estimation = {:.6f}".format(np.mean(I)),"Variance = {:.6f}".format(var))

###################################### PART C #################################
print('\nDifferent Monte Carlo Estimation for function 20+x^2+y^2-10(cos(2*pi*x)+cos(2*pi*y))\n')
# Define function
def integrand3(x,y):
    return 20 + x**2 + y**2 - 10*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))

#Define limites for integrals
a3 = -5
b3 = 5
# use N draws 
N= 1000
exactval=3666.67
def estimate3():
    x = np.random.uniform(low=a3, high=b3, size=N) # N values uniformly drawn from a to b 
    y = np.random.uniform(low=a3, high=b3, size=N) # N values uniformly drawn from a to b 
    z =integrand3(x,y)   # CALCULATE THE f(x) 
    V = b3-a3
    I = V * V * np.sum(z)/ N;
    return I

I = np.empty(50)
for i in range(50):        
    I[i] = estimate3()
var = np.var(I)
print("Monte Carlo estimation using Uniform Sampling")
print("Estimation = {:.6f}".format(np.mean(I)),"Variance = {:.6f}".format(var/10000))


