# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:50:31 2019

@author: lindsey
"""

import pandas as pd
import hdbscan 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster

def plot_clusters(data, algorithm, args, kwds):
    a=algorithm(*args, **kwds)
    a.fit_predict(data)
    palette = sns.color_palette('deep', np.unique(a.labels_).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in a.labels_]
    plt.scatter(data.T[0], data.T[1], c=colors)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('{} Clusters found by {} for wpp feb 28'.format(a.labels_.max(),str(algorithm.__name__), fontsize=12))
    return a


data=pd.read_csv("/home/lindsey/insightproject/clusterarray.csv")
X=np.array(data.x)
Y=np.array(data.y)
clusterer = hdbscan.HDBSCAN()
clusterer.fit(data)

data=np.array(list(zip(X,Y)))
plt.scatter(data[:,0],data[:,1],s=5)
plot_clusters(data, hdbscan.HDBSCAN,(),{'min_cluster_size':12,'min_samples':3})
