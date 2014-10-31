from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from sklearn import metrics
from sklearn.cluster import Ward
from sklearn.cluster import KMeans

# parameters #################################################################

start = datetime.now()


# Only loading one time
print "Step0: Loading raw data"
raw_train = pd.read_csv('Accounting_ReportsDump_20141030.csv', header=0)


features = set(raw_train)
train = raw_train

for i in features:
    if i != 'Combined Queries' and i != 'Report ID' and i != 'Object Name' and i != 'Report Name' and i != 'Operands':
        print i
        train = pd.concat([train, pd.get_dummies(raw_train[i])], axis=1)
        
freq = train.groupby('Report ID').sum()
freq = freq.drop('Has Combined Queries', 1)


# Train Model #############################

num_cluster = 12

kmean = KMeans(n_clusters=num_cluster, max_iter=400, verbose = 0, n_jobs = 2, n_init=20, tol=1e-6)
model_kmean = kmean.fit(freq)
        
ward = Ward(n_clusters=num_cluster)
model_ward = ward.fit(freq)


from sklearn.neighbors import kneighbors_graph
connectivity = kneighbors_graph(freq, n_neighbors=4)

#ward = Ward(n_clusters=num_cluster, connectivity = connectivity)
#model_ward = ward.fit(freq)

# Visualization #####################################################

import mpl_toolkits.mplot3d.axes3d as p3
import pylab as pl
from sklearn.datasets.samples_generator import make_friedman3

def plot(model, data, name):
    fig = pl.figure()
    ax = p3.Axes3D(fig)
    ax.set_xlabel('Result Object')
    ax.set_ylabel('DIMENSION')
    ax.set_zlabel('eBusiness Operations')
    label = model.labels_

    temp = pd.concat([data, pd.DataFrame(label, index=data.index, columns={'label'})], axis=1)
    
    for l in np.unique(label):
        ax.plot3D(temp.ix[temp['label']==l]['Result Object'].as_matrix(), 
                  temp.ix[temp['label']==l]['DIMENSION'].as_matrix(),
                  temp.ix[temp['label']==l]['eBusiness Operations'].as_matrix(),
                  'o', color=pl.cm.jet(float(l) / np.max(label + 1)))
    pl.title(name + ' Cluster visulization')
    
plot(kmean, freq, 'Kmean')
plot(ward, freq, 'Ward')

print "It takes time = ", datetime.now() - start
