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
#C1 = np.array([[1, 0], [0, 1]])
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])
# generate zero centered stretched Gaussian data
C2 = np.array([[2.0, -1.0], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C2) + np.array([2, 2])
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