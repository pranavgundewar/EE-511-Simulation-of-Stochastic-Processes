"""
@author: Pranav Gundewar
Project 3
Q3- Clustes of Text
Dataset: nips-87-92
"""
# Importing Libraries
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Reading the data using 
inputData = np.genfromtxt('nips-87-92.csv', delimiter=',')
data = inputData[1:,2:]
print('Data Size:',data.shape)

label = np.zeros(700, dtype=np.int)
label = inputData[1:,1]
k = []
distance = []
score = []

for i in range (2, 20):
    kmeans = KMeans(n_clusters = i, init='k-means++').fit(data)
    clusterLables = kmeans.labels_
    k.append(i)
    distance.append(kmeans.inertia_)
    a = metrics.silhouette_score(data, clusterLables, metric='euclidean')
    score.append(a)
    index = np.argmax(score)+2
    print ("For ",i," Clusters","\tSilhouette Coefficient: %0.3f"
      % metrics.silhouette_score(data, clusterLables, metric='euclidean'))

print('Optimal Value of k: ',index)

kNewMeans = KMeans(n_clusters = index,init='k-means++').fit(data)
newClusterLables = kNewMeans.labels_
class1 = []
class2 = []

for i in range (len(label)):
    if newClusterLables[i] == 1:
        class2.append(label[i])
    else:
        class1.append(label[i])
print('Document IDs for Cluster 1: \n',class1)
print('Document IDs for Cluster 2: \n',class2)        
plt.plot(k, distance)