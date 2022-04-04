import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(10000, 2)

def euclidean_distance(a,b):
    # dist = sqrt of feature vectors (a - b)
    # which can be written as a^2 - 2ab + b^2
    # which in term translates to dot product of a * a, dot product of b * a
    # dot product of b *b 
    return np.sqrt(np.dot(a,a) - 2 * np.dot(a,b) + np.dot(b,b))
    
# a = np.array([[0,0]])
# b = np.array([[1,1])]

def cosine_distance(a,b):
    # cosine distance can be calculated as 
    # k(x,y) = dot(x,y) / (||x|| * ||y||)
    divisor = (np.linalg.norm(a) * np.linalg.norm(b))
    if divisor == 0:
        return 0
    output = np.dot(a,b) / divisor
    
    if np.isnan(output):
        return 0
    return 1 - output

def jaccard_distance(a,b):
    # we calculate generalized jaccard distance 
    # by the( 1 - ratio of sumo of minimum and maximum)
    return 1 - (np.array([a,b]).min(axis=0).sum() / np.array([a,b]).max(axis=0).sum())


class K_Means:
    def __init__(self, k=2,  max_iter=300, distance_function=euclidean_distance):
        self.k = k
        self.max_iter = max_iter
        self.sse_total = 0
        self.distance_function = distance_function
        self.iter = 0
    def fit(self,x):
        self.clust_index = np.random.randint(len(x), size=self.k)
        self.cluster_centers = x[self.clust_index]
        self.classification = np.zeros(shape=len(x), dtype=[('index',int), ('cluster',int)])
        self.init_clusters = self.cluster_centers
        self.optimized = False
        new_cluster_centers = self.cluster_centers.copy()
        distances = np.zeros(shape=(len(x), self.k))
        
        for iter_ in range(self.max_iter):
            self.optimized = True
            self.iter = iter_
            self.sse_total = 0
            for i , t in enumerate(x):
                for j, center in enumerate(self.cluster_centers):
                    distances[i][j] = self.distance_function(t,center)
                self.classification[i] = (i, np.argmin(distances[i]))
                
            for i in range(self.k):
                new_cluster_centers[i] = np.average(x[self.classification[self.classification['cluster'] ==i]['index']],axis=0)
                clust_dist = self.distance_function(self.cluster_centers[i], new_cluster_centers[i])
                if clust_dist > 0:
                    self.optimized  = False
            self.cluster_centers = new_cluster_centers.copy()
            for i,cluster in enumerate(self.cluster_centers):
                sse_c = self.SSE_cluster(x[self.classification[self.classification['cluster'] ==i]['index']], cluster)
                self.sse_total += sse_c
            print("Iter", self.iter, "SSE ", self.sse_total )
            if self.optimized:
                break

    def SSE_cluster(self,x,c,distance_function = euclidean_distance):
        output = 0
        for t in x:
            output += distance_function(t,c) ** 2
        return output

#fs = X[model.classification[model.classification['cluster'] == k]['index']]

#print(fs)

model = K_Means(k=10,distance_function=euclidean_distance, max_iter=50)
model.fit(X)


colors = [
    'black', 'yellow', 'orange', 'aqua', 'white', 'red', 'green', 'blue', 'navy', 'purple', 'indigo'
]
for k in range(10):
    color = colors[k]
    fs = X[model.classification[model.classification['cluster'] == k]['index']]
    plt.scatter(fs[:, 0], fs[:, 1], color=color, s=2, linewidths=1)

for centroid in model.cluster_centers:
    plt.scatter(centroid[0], centroid[1],
                marker="x", color="k", s=150, linewidths=5)

plt.show()


model = K_Means(k=10,distance_function=jaccard_distance, max_iter=100)
model.fit(X)


colors = [
    'black', 'yellow', 'orange', 'aqua', 'white', 'red', 'green', 'blue', 'navy', 'purple', 'indigo'
]
for k in range(10):
    color = colors[k]
    fs = X[model.classification[model.classification['cluster'] == k]['index']]
    plt.scatter(fs[:, 0], fs[:, 1], marker="x", color=color, s=150, linewidths=5)
    
for centroid in model.cluster_centers:
    plt.scatter(centroid[0], centroid[1],
                marker="o", color="k", s=150, linewidths=5)
plt.show()


model = K_Means(k=10,distance_function=jaccard_distance, max_iter=10000)
model.fit(X)

colors = [
    'black', 'yellow', 'orange', 'aqua', 'white', 'red', 'green', 'blue', 'navy', 'purple', 'indigo'
]
for k in range(10):
    color = colors[k]
    fs = X[model.classification[model.classification['cluster'] == k]['index']]
    plt.scatter(fs[:, 0], fs[:, 1], marker="x", color=color, s=150, linewidths=5)

for centroid in model.cluster_centers:
    plt.scatter(centroid[0], centroid[1],
                marker="o", color="k", s=150, linewidths=5)
        
plt.show()

X = np.genfromtxt('data/data.csv',delimiter=',')
Y = np.genfromtxt('data/label.csv',delimiter=',')

euclidean_model = K_Means(k=10, distance_function=euclidean_distance, max_iter=300)
euclidean_model.fit(X)

cosine_model = K_Means(k=10, distance_function=cosine_distance, max_iter=300)
euclidean_model.fit(X)

jacard_model = K_Means(k=10,distance_function=jaccard_distance, max_iter=300)
jacard_model.fit(X)

for i in range(k):
    a = Y[model.classification[euclidean_model.classification['cluster'] == 0]['index']].astype(int)
    np.bincount(a).max() / np.bincount(a).sum()

np.bincount(Y[model.classification[model.classification['cluster'] == 1]['index']].astype(int))
