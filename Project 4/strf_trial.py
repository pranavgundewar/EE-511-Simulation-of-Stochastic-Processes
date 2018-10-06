from __future__ import division
import random
import numpy as np
import math
import warnings

warnings.simplefilter("error")

def integrateMC(func, dim, limit, N):
    I = 1. / N
    tol = []
    for n in range(dim):
        I *= (limit[n][1] - limit[n][0])

    for k in range(N):
        x = []
        for n in range(dim):
            x.append(random.uniform(limit[n][0], limit[n][1]))
        tol.append(func(x))
    est_var = 0
    m = np.mean(tol)
    for nums in tol:
        est_var += (nums - m) ** 2
    return I * np.sum(tol), est_var * I / (N - 1)


def SS(func, dim, limit, N, sigma):
    # for 2 dimension, it's has 2 part,
    # We divide X into [0,sigma] and [sigma,1], remain Y same
    I = 1. / N
    N_a = 0.
    N_b = 0.
    tol_a = []
    tol_b = []

    # We only deal with X

    for n in range(dim):
        if n==0:
            N_a = N * (limit[n][1] * sigma - limit[n][0])/(limit[n][1] - limit[n][0])
            N_b = N * (limit[n][1] - limit[n][1] * sigma) / (limit[n][1] - limit[n][0])


    for n in range(dim):
        I *= (limit[n][1] - limit[n][0])

    for k in range(int(N_a)):
        x = []
        for n in range(dim):
            if n == 0:
                x.append(random.uniform(limit[n][0], sigma * limit[n][1]))
            else:
                x.append(random.uniform(limit[n][0], limit[n][1]))

        tol_a.append(func(x))

    for k in range(int(N_b)):
        x = []
        for n in range(dim):
            if n == 0:
                x.append(random.uniform(sigma * limit[n][1], limit[n][1]))
            else:
                x.append(random.uniform(limit[n][0], limit[n][1]))

        tol_b.append(func(x))
    #print (tol_a)
    #print (np.var(tol_b))
    est_var = float(sigma - limit[0][0])/(limit[0][1] - limit[0][0]) / N * np.var(tol_a) + \
              float(limit[0][1] - sigma)/(limit[0][1] - limit[0][0]) / N * np.var(tol_b)
    val =  I * np.sum(tol_a) + I * np.sum(tol_b)
    diff = abs(np.var(tol_a) - np.var(tol_b))
    return val, est_var, diff

def SS_2(func, dim, limit, N, sigmax, sigmay):
    # for 2 dimension, it's has 4 part
    N_a = N * sigmax * sigmay
    N_b = N * sigmax * (1 - sigmay)
    N_c = N * (1 - sigmax) * sigmay
    N_d = N * (1 - sigmax) * (1 - sigmay)
    I_a = 1. / N_a
    I_b = 1. / N_b
    I_c = 1. / N_c
    I_d = 1. / N_d
    tol_a = []
    tol_b = []
    tol_c = []
    tol_d = []

    for k in range(int(N_a)):
        x = []
        for n in range(dim):
            if n == 0:
                x.append(random.uniform(0, sigmax))
            else:
                x.append(random.uniform(0, sigmay))
        # print 'x',x
        tol_a.append(func(x))

    for k in range(int(N_b)):
        x = []
        for n in range(dim):
            if n == 0:
                x.append(random.uniform(0, sigmax))
            else:
                x.append(random.uniform(sigmay, 1))
        # print 'x2',x
        tol_b.append(func(x))

    for k in range(int(N_c)):
        x = []
        for n in range(dim):
            if n == 0:
                x.append(random.uniform(sigmax, 1))
            else:
                x.append(random.uniform(0, sigmay))
        # print 'x2',x
        tol_c.append(func(x))

    for k in range(int(N_d)):
        x = []
        for n in range(dim):
            if n == 0:
                x.append(random.uniform(sigmax, 1))
            else:
                x.append(random.uniform(sigmay, 1))
        # print 'x2',x
        tol_d.append(func(x))
    est_var = sigmax * sigmay / N * np.var(tol_a) + (1 - sigmay) * sigmax / N * np.var(tol_b) + \
              (1 - sigmax) * sigmay / N * np.var(tol_c) + (1 - sigmay) * (1 - sigmax) / N * np.var(tol_d)

    val = sigmax * sigmay* I_a * np.sum(tol_a) + sigmax * (1-sigmay)* I_b * np.sum(tol_b) + \
          (1-sigmax)*sigmay*I_c * np.sum(tol_c) + (1-sigmax)*(1-sigmay)*I_d * np.sum(tol_d)
    diff = abs(np.var(tol_a) - np.var(tol_b)) + abs(np.var(tol_a) - np.var(tol_c)) + \
           abs(np.var(tol_a) - np.var(tol_d))
    return val, est_var, diff

def f_a(x):
    return np.exp(-x[0]**4-x[1]**4)

def f_b(x):
    return math.cos(math.pi + 5 * (x[0] + x[1]))

def f_c(x):
    """ integrand function """
    return (abs(4 * x[0] - 2) * abs(4 * x[1] - 2))
a2 = -1*np.pi
b2 = np.pi

dim = 2
limit_a = [[a2, b2], [a2, b2]]
limit_b = [[-1, 1], [-1, 1]]
limit_b1 = [[-1, 0], [-1, 1]]
limit_b2 = [[0, 1], [-1, 1]]
limit_c = [[0, 1], [0, 1]]
N = 1000  # starting value for number of MC samples

#***************** Function A Operation *************************

real_a = 3.28626
dict_a = {}
dict_a2 = {}
for sigma in np.arange(0.001, 1, 0.001):
    S_a = SS(f_a, dim, limit_b, N, sigma)
    #dict_a[S_a[2]] = S_a[0],S_a[1]
    dict_a[S_a[1]]= S_a[0] # find the minimum estimate variance
    dict_a2[abs(S_a[0] - real_a)] = S_a[0],S_a[1] # find the closed value
print ("*************Function A real value: ",real_a,"*****************")
a = (min(dict_a, key=float))
print ("For minimum estimate: ")
print ("Stratified integral value of a: ", dict_a[a] \
       ,"estimate variance of a: " , a )

a2 = (min(dict_a2, key=float))
print ("For closest value: ")
print ("Stratified integral value of a: ", dict_a2[a2][0]  \
       ,"estimate variance of a: " , dict_a2[a2][1] )


v_a = integrateMC(f_a, dim, limit_a, N)
print ("MC integral a:", v_a[0], "MC estimate variance of a: ",v_a[1])

