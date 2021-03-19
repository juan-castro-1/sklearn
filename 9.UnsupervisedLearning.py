#%%
'CLUSTERING'
'K-MEANS'
from sklearn import datasets
from sklearn import pipeline

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
X.shape
y.shape
X.shape

# %%
from sklearn.cluster import KMeans
k = 5
kmeans_ = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)
#%%
y_pred
y_pred is kmeans.labels_
kmeans.cluster_centers_
kmeans.n_clusters
#%%
import numpy as np
X_new = np.array([[0,2], [3,2], [-3,3],[-3,2.5]])
kmeans.predict(X_new)

#%%
kmeans.transform(X_new)
'measures the distance from each insntance to every centroid'
#%%
'como sabemos que es min global?? aca hacemos Kmeans n_init veces'
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1)
'sklearn elige el mejor: el de menor de distancia cuadrada'
#%%
'K-Means++'
'''
The KMeans class actually uses this initialization method by default. If you want to
force it to use the original method (i.e., picking k instances randomly to define the
initial centroids), then you can set the init hyperparameter to "random". You will
rarely need to do this.
'''
'accelerated K-Means and Mini-batch K-means'
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5)
minibatch_kmeans.fit(X)
#%%
'''
eleccion de clusters:
-elbow rule
el k que hace caer mas la inertia

-silhouette score
ver libro p248
'''
from sklearn.metrics import silhouette_score
silhouette_score(X, kmeans.labels_)
#%%
'''
K-Means does not behave very well when the clusters have varying sizes,
different densities, or non-spherical shapes

It is important to scale the input features before you run K-Means,
or else the clusters may be very stretched, and K-Means will per‐
form poorly. Scaling the features does not guarantee that all the
clusters will be nice and spherical, but it generally improves things.
'''
#%%%
from sklearn.datasets import load_iris

X_digits, y_digits = load_iris(return_X_y=True)
#%%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

#%%
log_reg.score(X_test, y_test)
log_reg.fit(X_train, y_train)
#%%
log_reg.score(X_test,y_test)
#%%
'''
Let’s see if we can do better by using KMeans as a preprocessing step. We will create a pipeline that will first cluster the
training set into 50 clusters and replace the images with their distances to these 50
clusters, then apply a logistic regression model.
'''

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

pipeline = Pipeline([
    ('kmeans', KMeans(n_clusters=50)),
    ('log_reg', LogisticRegression())
])
#%%
pipeline.fit(X_train, y_train)
#%%
pipeline.score(X_test, y_test)
#%%
'''
the best value of k is simply the one that results in the best
classification performance during cross-validation
'''
from sklearn.model_selection import GridSearchCV

param_grid = dict(kmeans__n_clusters=range(2,100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)
#%%
grid_clf.best_params_
#%% 
grid_clf.score(
    X_test, y_test
)
#%%
'SEMI-SUPERVISED LEARNING'
n_labeled = 50
log_reg = LogisticRegression()
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])

#%%
log_reg.score(X_test, y_test)
#%%
import numpy as np
k = 50
kmeans = KMeans(n_clusters=k)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digits_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digits_idx]
#%%
y_representative_digits = np.array([
    4,8,0,6,8,3,7,7,9,2,
    5,5,8,5,2,1,2,9,6,1,
    1,6,9,0,8,3,0,7,4,1,
    6,5,2,4,1,8,6,3,9,2,
    5,2,9,4,7,6,2,3,1,1
])
#%%
log_reg = LogisticRegression()
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test)
#%%
'DBSCAN'
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.05)
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan.fit(X)
#%%
dbscan.labels_
#%%
len(dbscan.core_sample_indices_)
#%%
dbscan.core_sample_indices_
#%%
dbscan.components_
#%%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
#%%
import numpy as np
X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
knn.predict(X_new)
#%%
knn.predict_proba(X_new)
#%%
y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
y_pred[y_dist > 0.2] = -1
y_pred.ravel()

'''
In short, DBSCAN is a very simple yet powerful algorithm, capable of identifying any
number of clusters, of any shape, it is robust to outliers, and it has just two hyper‐
parameters (eps and min_samples). However, if the density varies significantly across
the clusters, it can be impossible for it to capture all the clusters properly. Moreover,
its computational complexity is roughly O(m log m), making it pretty close to linear
with regards to the number of instances. However, Scikit-Learn’s implementation can
require up to O(m2
) memory if eps is large
'''
#%%
'GAUSSIAN MIXTURES (GMM)'
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)
#%%
gm.weights_
gm.means_
gm.covariances_
#%%
gm.converged_
gm.n_init
#%%
gm.predict_proba(X)
gm.predict_proba(X)
#%5
X_new, y_new = gm.sample(6)
X_new
#%%
y_new
gm.score_samples(X)
#%%
'para outliers'
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]
#%%
'selecting clusters'
gm.bic(X)
gm.aic(X)
#%%
'Bayesian Gaussian Mix Models'
from sklearn.mixture import BayesianGaussianMixture
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)
np.round(bgm.weights_,2)
#%%


