# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:03:47 2021

@author: juan_
"""

'TRAINING AND VIS TREE'
#%%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target


tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

from sklearn.tree import export_graphviz

export_graphviz(
    tree_clf,
    out_file=image_path("iris_tree.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
 )

## probabilities

tree_clf.predict_proba([[5,1.5]])
tree_clf.predict([[5,1.5]])

#%%
'reg'

from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

tree_reg = DecisionTreeRegressor(max_depth=2)
abc = tree_reg.fit(X, y)

tree.plot_tree(abc)
tree.plot_tree(abc,
               filled=True)

#%%

from sklearn.datasets import load_iris
from sklearn import  tree

X, y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

tree.plot_tree(clf)

#%%

import graphviz

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render('C:\\Users\\juan_\\Dropbox\\Mi PC (LAPTOP-H9MAOJRB)\\Desktop\\Py Projects\\Sklearn_book\\iris_tree')

dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=iris.feature_names,  
                      class_names=iris.target_names,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 






