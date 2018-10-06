"""
@author: Pranav Gundewar
Project #5: Optimization & Sampling via MCMC
Q2- MCMC for Optimization
"""
#%%
# Importing Libraries
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import benchmarks

global str
X = 500
Y = 500

alpha = 1.1
def schwefel(x,y):
    value = 418.9829 * 2 - x * math.sin(math.sqrt(abs(x))) - y * math.sin(math.sqrt(abs(y)))
    return value

value = []
N = 100
MarkovLength = 100 #Iteration time
final_x = []
final_y = []

for _ in range(N):
    Step = 0.1 
    Temperature = 1e5
    tol = 1e-5
    iteration = 0

    route_x = []
    route_y = []
    rnd = random.random()
    OldX = -X * random.random()
    OldY = -Y * random.random()

    BestX = OldX
    BestY = OldY

    route_x.append(BestX)
    route_y.append(BestY)
    while Temperature > tol:

        # Choosing the Cooling style
        #Temperature = (0.88 ** (iteration + 1) ) * Temperature #Exponential
#        Temperature = Temperature / ( 1 + alpha * math.log( 1 + iteration,math.e)) #logarithmic
        Temperature = Temperature / (1 + 0.15*iteration) # polynomial
        accept = 0.0
        i = 0
        while i < MarkovLength :
            p = 0 # Choose the right range value
            while p == 0:
                NewX = OldX + Step * X * (random.random() - 0.5)
                NewY = OldY + Step * Y * (random.random() - 0.5)
                if((NewX >= -X) and (NewX <= X) and (NewY >= -Y) and (NewY <= Y)):
                    p = 1

            if(schwefel(BestX,BestY) > schwefel(NewX,NewY)):
                #Preserve the pervious best solution
                PreBestX = BestX
                PreBestY = BestY

                #Update the best solution
                BestX = NewX
                BestY = NewY
                route_x.append(BestX)
                route_y.append(BestY)

                #Metropolis Process
            if (schwefel(OldX,OldY) - schwefel(NewX,NewY) > 0):
                OldX = NewX
                OldY = NewY
                accept += 1
            else:
                changer = -1. * (schwefel(NewX,NewY) - schwefel(OldX,OldY))/Temperature
                rnd = random.random()
                p1 = math.exp(changer)
                if p1 > rnd:
                    OldX = NewX
                    OldY = NewY
                    accept += 1

            i = i + 1
        iteration += 1

    value.append(schwefel(BestX,BestY))
    final_x.append(route_x[:])
    final_y.append(route_y[:])

#*************** Question 3.1 and 3.4 ******************
index = []
for i in range(N):
    if value[i] < 0.1:
        index.append(i)
print (index)
key = {}
#%%
#it belongs to minimum value, then find minimum route length
for i in index:
    key[i] = len(final_y[i])
print (key)
#%%
a = min(key, key=key.get)
#k = len(final_y[a])
X = np.arange(-500, 500, 10)
Y = np.arange(-500, 500, 10)
X, Y = np.meshgrid(X, Y)
Z = np.zeros(X.shape)
def schwefel_arg0(sol):
    return benchmarks.schwefel(sol)[0]
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = schwefel_arg0((X[i, j], Y[i, j]))
plt.figure()
contour = plt.contour(X, Y, Z)
plt.colorbar(contour)
plt.title('Scwefel Contours Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(final_x[a],final_y[a])
plt.scatter(final_x[a],final_y[a])
#plt.plot(final_x[a][0:int (k/3)],final_y[a][0:int(k/3)],'ow')
#plt.plot(final_x[a][int(k/3):int( 2 * k/3)],final_y[a][int(k/3):int( 2 * k/3)],'^b')
#plt.plot(final_x[a][int(2 * k/3):],final_y[a][int(2 * k/3):],'*k')
plt.show()
#********************************************************
#%%

#*************** Question 3.3 Start **************************
plt.figure()
axes = plt.gca()
plt.hist(value,bins = 40)
plt.xlabel('Converge minimum value')
plt.ylabel('Frequency of corresponding minimum value')
#str = "Polynomial Cooling Schedule with "+str(MarkovLength)+ " Iterations"
str = "Logarithmic Cooling Schedule with "+str(MarkovLength)+ " Iterations"
#str = "Exponential Cooling Schedule with "+str(MarkovLength)+ " Iterations"
plt.title(str)
plt.show()

#*************** Question 3.3 End **************************


