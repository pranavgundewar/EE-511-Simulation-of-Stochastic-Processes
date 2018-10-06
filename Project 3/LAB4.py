"""
@author: Pranav Gundewar
Project 3
Q2 Part - EM Clustering
Dataset: old faithful
"""
# =============================================================================

import numpy as np
import math, random, copy
import sys
import matplotlib.pyplot as plt
from sklearn import mixture

def mixGaussian(n,mean,cov,weights):
    '''
    The input of function must be numpy arrays.
    Here only the case where each component has the same covariance is implemented
    '''
    assert sum(weights)==1
    category=np.flatnonzero(np.random.multinomial(1,weights))
    cmean=mean[category,:][0]
    return np.random.multivariate_normal(cmean,cov)


# ============================================================================
def expectation_maximization(t, nbclusters=2, nbiter=3, normalize=False,\
        epsilon=0.001, monotony=False, datasetinit=True):
    """ 
    Each row of t is an observation, each column is a feature 
    'nbclusters' is the number of seeds and so of clusters
    'nbiter' is the number of iterations
    'epsilon' is the convergence bound/criterium
    Overview of the algorithm:
    -> Draw nbclusters sets of (μ, σ, P_{#cluster}) at random (Gaussian 
       Mixture) [P(Cluster=0) = P_0 = (1/n).∑_{obs} P(Cluster=0|obs)]
    -> Compute P(Cluster|obs) for each obs, this is:
    [E] P(Cluster=0|obs)^t = P(obs|Cluster=0)*P(Cluster=0)^t
    -> Recalculate the mixture parameters with the new estimate
    [M] * P(Cluster=0)^{t+1} = (1/n).∑_{obs} P(Cluster=0|obs)
        * μ^{t+1}_0 = ∑_{obs} obs.P(Cluster=0|obs) / P_0
        * σ^{t+1}_0 = ∑_{obs} P(Cluster=0|obs)(obs-μ^{t+1}_0)^2 / P_0
    -> Compute E_t=∑_{obs} log(P(obs)^t)
       Repeat Steps 2 and 3 until |E_t - E_{t-1}| < ε
    """
    def pnorm(x, m, s):
        """ 
        Compute the multivariate normal distribution with values vector x,
        mean vector m, sigma (variances/covariances) matrix s
        """
        xmt = np.matrix(x-m).transpose()
        for i in range(len(s)):
            if s[i,i] <= sys.float_info[3]: # min float
                s[i,i] = sys.float_info[3]
        sinv = np.linalg.inv(s)
        xm = np.matrix(x-m)
        return (2.0*math.pi)**(-len(x)/2.0)*(1.0/math.sqrt(np.linalg.det(s)))\
                *math.exp(-0.5*(xm*sinv*xmt))

    def draw_params():
            if datasetinit:
                tmpmu = np.array([1.0*t[random.uniform(0,nbobs),:]],np.float64)
            else:
                tmpmu = np.array([random.uniform(min_max[f][0], min_max[f][1])\
                        for f in range(nbfeatures)], np.float64)
            return {'mu': tmpmu,\
                    'sigma': np.matrix(np.diag(\
                    [(min_max[f][1]-min_max[f][0])/2.0\
                    for f in range(nbfeatures)])),\
                    'proba': 1.0/nbclusters}

    nbobs = t.shape[0]
    nbfeatures = t.shape[1]
    min_max = []
    # find xranges for each features
    for f in range(nbfeatures):
        min_max.append((t[:,f].min(), t[:,f].max()))
    
    ### Normalization
    if normalize:
        for f in range(nbfeatures):
            t[:,f] -= min_max[f][0]
            t[:,f] /= (min_max[f][1]-min_max[f][0])
    min_max = []
    for f in range(nbfeatures):
        min_max.append((t[:,f].min(), t[:,f].max()))
    ### /Normalization

    result = {}
    quality = 0.0 # sum of the means of the distances to centroids
    random.seed()
    Pclust = np.ndarray([nbobs,nbclusters], np.float64) # P(clust|obs)
    Px = np.ndarray([nbobs,nbclusters], np.float64) # P(obs|clust)
    # iterate nbiter times searching for the best "quality" clustering
    for iteration in range(nbiter):
        ##############################################
        # Step 1: draw nbclusters sets of parameters #
        ##############################################
        params = [draw_params() for c in range(nbclusters)]
        old_log_estimate = sys.maxsize         # init, not true/real
        log_estimate = sys.maxsize/2 + epsilon # init, not true/real
        estimation_round = 0
        # Iterate until convergence (EM is monotone) <=> < epsilon variation
        while (abs(log_estimate - old_log_estimate) > epsilon\
                and (not monotony or log_estimate < old_log_estimate)):
            restart = False
            old_log_estimate = log_estimate
            ########################################################
            # Step 2: compute P(Cluster|obs) for each observations #
            ########################################################
            for o in range(nbobs):
                for c in range(nbclusters):
                    # Px[o,c] = P(x|c)
                    Px[o,c] = pnorm(t[o,:],\
                            params[c]['mu'], params[c]['sigma'])
            #for o in xrange(nbobs):
            #    Px[o,:] /= math.fsum(Px[o,:])
            for o in range(nbobs):
                for c in range(nbclusters):
                    # Pclust[o,c] = P(c|x)
                    Pclust[o,c] = Px[o,c]*params[c]['proba']
            #    assert math.fsum(Px[o,:]) >= 0.99 and\
            #            math.fsum(Px[o,:]) <= 1.01
            for o in range(nbobs):
                tmpSum = 0.0
                for c in range(nbclusters):
                    tmpSum += params[c]['proba']*Px[o,c]
                Pclust[o,:] /= tmpSum
                #assert math.fsum(Pclust[:,c]) >= 0.99 and\
                #        math.fsum(Pclust[:,c]) <= 1.01
            ###########################################################
            # Step 3: update the parameters (sets {mu, sigma, proba}) #
            ###########################################################
            #### print ("iter:", iteration)
            #### print ("estimation#: ", estimation_round)
            #### print ("params: ", params)
            for c in range(nbclusters):
                tmpSum = math.fsum(Pclust[:,c])
                params[c]['proba'] = tmpSum/nbobs
                if params[c]['proba'] <= 1.0/nbobs:           # restart if all
                    restart = True                             # converges to
                    print ("Restarting, p:",params[c]['proba']) # one cluster
                    break
                m = np.zeros(nbfeatures, np.float64)
                for o in range(nbobs):
                    m += t[o,:]*Pclust[o,c]
                params[c]['mu'] = m/tmpSum
                s = np.matrix(np.diag(np.zeros(nbfeatures, np.float64)))
                for o in range(nbobs):
                    s += Pclust[o,c]*(np.matrix(t[o,:]-params[c]['mu']).transpose()*\
                            np.matrix(t[o,:]-params[c]['mu']))
                params[c]['sigma'] = s/tmpSum
                #### print ("------------------")
                #### print (params[c]['sigma'])

            ### Test bound conditions and restart consequently if needed
            if not restart:
                restart = True
                for c in range(1,nbclusters):
                    if not np.allclose(params[c]['mu'], params[c-1]['mu'])\
                    or not np.allclose(params[c]['sigma'], params[c-1]['sigma']):
                        restart = False
                        break
            if restart:                # restart if all converges to only
                old_log_estimate = sys.maxsize         # init, not true/real
                log_estimate = sys.maxsize/2 + epsilon # init, not true/real
                params = [draw_params() for c in range(nbclusters)]
                continue
            ### /Test bound conditions and restart

            ####################################
            # Step 4: compute the log estimate #
            ####################################
            log_estimate = math.fsum([math.log(math.fsum(\
                    [Px[o,c]*params[c]['proba'] for c in range(nbclusters)]))\
                    for o in range(nbobs)])
            #### print ("(EM) old and new log estimate: ", old_log_estimate, log_estimate)
            estimation_round += 1

        # Pick/save the best clustering as the final result
        quality = -log_estimate
        if not quality in result or quality > result['quality']:
            result['quality'] = quality
            result['params'] = copy.deepcopy(params)
            result['clusters'] = [[o for o in range(nbobs)\
                    if Px[o,c] == max(Px[o,:])]\
                    for c in range(nbclusters)]
    return result
# ============================================================================
n=2
mean=np.array([[-4,0],[4,0]])
cov=np.array([[1,0],[0,1]])
weights=np.array([0.5,0.5])


x = mixGaussian(n,mean,cov,weights)
x = np.append([x],  [mixGaussian(n,mean,cov,weights)], axis=0)
for var in range(298):
    x =  np.append(x, [mixGaussian(n,mean,cov,weights)], axis=0)
result0 = expectation_maximization(x)
param = result0['params']
index = result0['clusters']
first_group = index[0]
second_group = index[1]
u = x[first_group[0]]
u = np.append([u], [x[first_group[1]]], axis = 0)
for var in range (2,len(first_group)):
    u = np.append(u, [x[first_group[var]]], axis = 0)
v = x[second_group[0]]
v = np.append([v], [x[second_group[1]]], axis = 0)
for var in range (2,len(second_group)):
    v = np.append(v, [x[second_group[var]]], axis = 0)
first = param[0]
second = param[1]
print(param)
plt.scatter(first['mu'][0], first['mu'][1], s = 100, c = 'r', marker='x', linewidth='3')
plt.scatter(second['mu'][0], second['mu'][1], s = 100, c = 'black', marker='x', linewidth='3')
plt.scatter(u[:,0], u[:,1], c = 'red', alpha=0.2)
plt.scatter(v[:,0], v[:,1], c = 'black', alpha=0.2)
plt.show()
# ============================================================================   
temp0 = []
temp1 = []
temp2 = []         
with open('old-faithful.txt') as f:
    for line in f:
        line = line.split() # to deal with blank 
        if line:            # lines (ie skip them)
            line = [float(i) for i in line]
            line.pop(0)
            temp0.append(line[0])
            temp1.append(line[1])
            temp2.append(line)

duration = np.asarray(temp0)
time = np.asarray(temp1)
data = np.append([duration], [time], axis = 0)
data = data.transpose()


result1 = expectation_maximization(data)
param1 = result1['params']
first1 = param1[0]
second1 = param1[1]

index1 = result1['clusters']
first_group1 = index1[0]
second_group1 = index1[1]
i = data[first_group1[0]]
i = np.append([i], [data[first_group1[1]]], axis = 0)
for var in range (2,len(first_group1)):
    i = np.append(i, [data[first_group1[var]]], axis = 0)
j = data[second_group1[0]]
j = np.append([j], [data[second_group1[1]]], axis = 0)
for var in range (2,len(second_group1)):
    j = np.append(j, [data[second_group1[var]]], axis = 0)


clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(data)

x = np.linspace(1, 5)
y = np.linspace(40, 100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z)
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(i[:,0], i[:,1], c = 'red', alpha=0.2)
plt.scatter(j[:,0], j[:,1], c = 'black', alpha=0.2)
plt.scatter(first1['mu'][0], first1['mu'][1], s = 100, c = 'r', marker='x', linewidth='3')
plt.scatter(second1['mu'][0], second1['mu'][1], s = 100, c = 'black', marker='x', linewidth='3')
plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
print (param1)
