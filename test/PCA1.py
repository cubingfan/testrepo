# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:06:25 2018

@author: Sanjit Dasgupta
"""


from collections import defaultdict
import scipy
import traceback
import os
import numpy as np
import scipy
import scipy.stats
import scipy.io
import matplotlib.pyplot as plt
import sys
import seaborn as sns; sns.set()  # for plot styling

import sklearn.cluster
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#LOAD DATA
input_path = 'filtered_gene_bc_matrices/zv10_gtf89_cloche_gfp/'

if os.path.isfile(input_path + '/matrix.npz'):
    E = scipy.sparse.load_npz(input_path + '/matrix.npz')
else:
    E = scipy.io.mmread(input_path + '/matrix.mtx').T.tocsc()
    scipy.sparse.save_npz(input_path + '/matrix.npz', E, compressed=True)

print(E.shape)

#x = E[:,    :]
#x = StandardScaler().fit_transform(x)

X = E.toarray()
X = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

pca = PCA(n_components=20)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)

#kmeans = KMeans(n_clusters=4)
#kmeans.fit(X_pca)
#y_kmeans = kmeans.predict(X_pca)

X_new = pca.inverse_transform(X_pca)
#plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8,c=y_kmeans, cmap='tab10')
plt.axis('equal');