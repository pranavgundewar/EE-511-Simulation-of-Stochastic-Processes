"""
@author: Pranav Gundewar
Project 3
Q1- K Means Clustering
Dataset: old faithful
"""
# Importing Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# Reading the data using 
data = pd.read_table('old_faithful_data.txt', delim_whitespace = True)
print('Data Size:',data.shape)
f1 = data['d2'].values
f2 = data['d3'].values
X = np.array(list(zip(f1, f2)))

# Number of clusters
kmeans = KMeans(n_clusters=2,n_init=1,init='random',max_iter = 500)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_

#Plotting the scatter plot
colors = ['g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range (2):
    points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
plt.scatter(centroids[:,0],centroids[:,1], marker='*', s=200, c='red')
plt.title('K Means Clustering for old_faithful dataset')

print('Centroids for 2 clusters: \n',centroids)

