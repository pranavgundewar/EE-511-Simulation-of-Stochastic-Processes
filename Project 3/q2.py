"""
@author: Pranav Gundewar
Project 3
Q2 Part A - EM Clustering
Dataset: old faithful
"""
from scipy.stats import multivariate_normal as mvn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import pandas as pd


# Part A- 2-dimensional RNG for a Gaussian mixture model (GMM) pdf with 2 subpopulations
n_samples = 300
# generate random sample, two components
np.random.seed(10)
# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])
# generate zero centered stretched Gaussian data
C = np.array([[2.0, -1.0], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])
# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)
# display predicted scores by the model as a contour plot
x = np.linspace(-20., 40.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()


def em_gmm_orig(xs, pis, mus, sigmas, tol=0.01, max_iter=300):
#    data, label = xs
#    xs = np.asarray(data)
    n, p = xs.shape
    k = len(pis)

    ll_old = 0
    for l in range(max_iter):
        exp_A = []
        exp_B = []
        ll_new = 0

        # E-step
        ws = np.zeros((k, n))
        for j in range(len(mus)):
            for i in range(n):
                ws[j, i] = pis[j] * mvn(mus[j], sigmas[j]).pdf(xs[i])
        ws /= ws.sum(0)
        
        # M-step
        pis = np.zeros(k)
        for j in range(len(mus)):
            for i in range(n):
                pis[j] += ws[j, i]
        pis /= n

        mus = np.zeros((k, p))
        for j in range(k):
            for i in range(n):
                mus[j] += ws[j, i] * xs[i]
            mus[j] /= ws[j, :].sum()
        print(l,' ',mus)
        sigmas = np.zeros((k, p, p))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(xs[i]- mus[j], (2,1))
                sigmas[j] += ws[j, i] * np.dot(ys, ys.T)
            sigmas[j] /= ws[j,:].sum()

        # update complete log likelihoood
        ll_new = 0.0
        for i in range(n):
            s = 0
            for j in range(k):
                s += pis[j] * mvn(mus[j], sigmas[j]).pdf(xs[i])
            ll_new += np.log(s)

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    return ll_new, pis, mus, sigmas

np.random.seed(12)

#create data set
n = 300
#_mus = np.array([[0,4], [-2,0]])
#_sigmas = np.array([[[3, 0], [0, 0.5]], [[1,0],[0,2]]])
#_pis = np.array([0.6, 0.4])
#xs = np.concatenate([np.random.multivariate_normal(mu, sigma, int(pi*n))
#                    for pi, mu, sigma in zip(_pis, _mus, _sigmas)])
#c = 2
xs = clf.sample((n))
data, label = xs
xs = np.asarray(data)
#ric = np.random.random((300,2))
#n, p = xs.shape
#k = len(ric)
#ric[:,1] = 1 - ric[:,0]
#m = np.empty(2)
#for i in range(len(ric)):
#    m[0] += ric[i,0]
#    m[1] += ric[i,1]
#print(m)
#print(m[0]+m[1])
#pic = np.empty(c)
#pic[0] = m[0] / (m[0]+m[1])
#pic[1] = m[1] / (m[0]+m[1])
#print(pic)
#mus = np.zeros((2, 2))
#for j in range(2):
#    for i in range(n):
#        mus[j] += ric[j, i] * xs[i]
#    mus[j] /= ric[j, :].sum()



pis = np.random.random(2)
pis /= pis.sum()
mus = np.random.random((2,2))
sigmas = np.array([np.eye(2)] * 2)
ll1, pis1, mus1, sigmas1 = em_gmm_orig(xs, pis, mus, sigmas)
#data, label = xs
#xs = np.asarray(data)

intervals = 101
ys = np.linspace(-8,8,intervals)
X, Y = np.meshgrid(ys, ys)
_ys = np.vstack([X.ravel(), Y.ravel()]).T

z = np.zeros(len(_ys))
for pi, mu, sigma in zip(pis1, mus1, sigmas1):
    z += pi*mvn(mu, sigma).pdf(_ys)
z = z.reshape((intervals, intervals))

ax = plt.subplot(111)
plt.scatter(xs[:,0], xs[:,1], alpha=0.2)
plt.contour(X, Y, z, N=10)
plt.axis([-8,6,-6,8])
ax.axes.set_aspect('equal')
plt.tight_layout()

