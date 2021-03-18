#%%
'PCA'
'''It seems reasonable to select the axis that preserves the maximum amount of var‐
iance, as it will most likely lose less information than the other projections. Another
way to justify this choice is that it is the axis that minimizes the mean squared dis‐
tance between the original dataset and its projection onto that axis. This is the rather
simple idea behind PCA
'''

'''
PCA assumes that the dataset is centered around the origin. As we
will see, Scikit-Learn’s PCA classes take care of centering the data
for you. However, if you implement PCA yourself (as in the pre‐
ceding example), or if you use other libraries, don’t forget to center
the data first.
'''
from os import replace
from numpy.core.fromnumeric import cumsum
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())

X, y = mnist['data'], mnist['target']
X.shape
y.shape

import pandas as pd
X = X.sample(frac=0.08, replace=True, random_state=1)
X.shape

#%%

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
#X2D.components_
pca.components_

#%%
'variance ratio'
pca.explained_variance_ratio_

'''
This tells you that 84.2% of the dataset’s variance lies along the first axis, and 14.6%
lies along the second axis
'''
import numpy as np

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1 

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
X_reduced.shape
X.shape

#%%
'PCA for compression'
X_reduced.shape
X.shape
# %%
'inversa'
pca = PCA(n_components=154)
X_reduced = pca.fit_transform(X)
X_recovered = pca.inverse_transform(X_reduced)
X_recovered.shape
# %%
'randomized PCA'
rnd_pca = PCA(n_components=154, svd_solver='randomized')
X_reduced = rnd_pca.fit_transform(X)
X_reduced.shape 

#%%
'''
One problem with the preceding implementations of PCA is that they require the
whole training set to fit in memory in order for the algorithm to run. Fortunately,
Incremental PCA (IPCA) algorithms have been developed: you can split the training
set into mini-batches and feed an IPCA algorithm one mini-batch at a time
'''

from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X, n_batches):
    inc_pca.partial_fit(X_batch)

'partial fit, NO fit'
X_reduced = inc_pca.transform(X)
X_reduced.shape

#%%
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)

#%% 
'kernel PCA, kPCA'
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)
X_reduced.shape
#%%
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
import numpy as np

clf = Pipeline([
    ('kpca', KernelPCA(n_components=2)),
    ('log_reg', LogisticRegression())
])

param_grid = [{
    'kpca__gamma': np.linspace(0.03,0.05,10),
    'kpca__kernel': ['rbf', 'sigmoid']
}]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X,y)
print(grid_search.best_params_)
#%%
'LOCAL LINEAR EMBEDDING, (LLE)'
'''
 It is a Manifold Learning technique that does not rely
on projections like the previous algorithms. In a nutshell, LLE works by first measur‐
ing how each training instance linearly relates to its closest neighbors (c.n.), and then
looking for a low-dimensional representation of the training set where these local
relationships are best preserved (more details shortly)
'''
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_neighbors=10,n_components=2)
X_reduced = lle.fit_transform(X)
X_reduced.shape

