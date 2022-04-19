#!/usr/bin/env python
# coding: utf-8

# # David Brookes
# # April 2022

# # K Nearest Neighbour algorithm
# # Task achieved using:
# # 1. Simple python commands
# # 2. Library software

# In[1]:


import numpy as np

import collections as colls

def euclidian_distance(x1, x2):
    return np.sqrt((np.sum(x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y): # Notice this is the same form as the scikit-learn function.
        self.X_train = X
        self.y_train = y
    
    def predict(self, X): # Notice this is the same form as the scikit-learn function.
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        # Compute distances.
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
        
        # Get k nearest samples and labels.
        k_indices = np.argsort(distances)[:self.k] # Sort in ascending order, and select the first k.
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        
        # Take the majority vote i.e select the most common class label.
        most_common = colls.Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

