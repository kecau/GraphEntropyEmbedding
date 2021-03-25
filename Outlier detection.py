#Importing the packages#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import copy
from sklearn import linear_model
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import heapq
from matplotlib.patches import Ellipse, Circle
import seaborn as sns

#Loading data#

def Loadingdata(City):
    X = pd.read_csv('EV'+City+'.csv',header = None)
    return X

#Loading the embedding vectors of Beijing#
X = Loadingdata(BJ)

pca = PCA(n_components = 2) #For visulization#
x_pca = pca.fit_transform(X)

pca1 = PCA(n_components = 1) #For applying to outlier detection by using box-plot#
x_pca1 = pca1.fit_transform(X)

#Visulization of the embedding space#
plt.scatter(x_pca[:,0],x_pca[:,1])

#Outier detection by using the local outlier factors#

lof = LocalOutlierFactor(n_neighbors = int(X.shape[0]/10), contamination = 0.1,algorithm = 'auto',n_jobs = -1,novelty= True)
lof.fit(X)
y_pred_outliers_LOF = lof.predict(X)

#Isolation Forest#
rng = np.random.RandomState(30)
IF = IsolationForest(behaviour = 'new', max_samples = int(X.shape[0]/10), random_state = rng, contamination = 'auto')
IF.fit(X)
y_pred_outliers_IF = IF.predict(X)

#Box-plot#

bt = plt.boxplot(x_pca1)
ol = bt['fliers'][0].get_ydata()

#Making ground truth#

#Calcuating the center of the embedding vectors#

cen = np.array(([np.mean(x_pca[:,0]),np.mean(x_pca[:,1])]))

#Defining the distance function#
def ed(m, n):
    return np.sqrt(np.sum((m - n) ** 2))

#Calculating the distance from each data point to the center#
dis = []
for i in range(x_pca.shape[0]):
    dis.append(ed(x_pca[i],cen))
dis = np.array(dis)

#Selecting 10% outliers#

top = heapq.nlargest(int(X.shape[0]/10), range(len(dis)), dis.take)
top = np.array(top).reshape(1,-1)
r = ed(x_pca[top[-1]],cen)
gd = []
c = 0
for i in range(dis.shape[0]):
    if dis[i] > r:
        gd.append(-1)
    else:
        gd.append(1)
gd = np.array(gd)

label = []
c = 0
for i in range(x_pca1.shape[0]):
    if x_pca1[i] in ol:
        label.append(-1)
    elif x_pca1[i] not in ol :
        label.append(1)
label = np.array(label)
label.shape

#Saving the results#
X['Label'] = y_pred_outliers_IF
X['Groud'] = gd
X.head()
X.to_csv('ODBJIF.csv')

X['Label'] = y_pred_outliers_LOF
X['Groud'] = gd
X.head()
X.to_csv('ODBJLOF.csv')

X['Label'] = label
X['Groud'] = gd
X.head()
X.to_csv('ODBJBP.csv')