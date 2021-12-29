# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 22:18:17 2021

@author: 17426
"""

import numpy as np
import umap
import sklearn.datasets
import umap.plot

# retrive mnist dataset
pendigits = sklearn.datasets.load_digits()
np.savetxt("mnist_data.txt",pendigits.data,delimiter=" ")
mnist_label = pendigits.target
np.savetxt("mnist_label.csv",mnist_label,delimiter=',')

# perform umap using umap module (defult parameter: n-neighbors=15, min-dist=0.1, e-pochs=200)
mapper = umap.UMAP().fit(pendigits.data)
umap.plot.points(mapper)
mnist_coordinates = mapper.embedding_
np.savetxt("mnist_module_output.csv",mnist_coordinates,delimiter=",")
# dimentionality reduction results are plotted using ggplot2 in R

# calculate the accuracy
from collections import Counter 
import pandas as pd
from sklearn import metrics
import matplotlib as plt

# read umap output (the first 2 columns should be x and y coordinates respectively)
data = pd.read_csv("mnist_module_output.csv",header=None)
coordinates = data.iloc[:,:2].to_numpy()
label = np.array(list(data.iloc[:,2]))

class KNN:
	def __init__(self, data, label):
		self.data = data
		self.label = label

	def distance(self, p1, p2):
		assert(len(p1) == len(p2))
		distance = 0
		for i,j in zip(p1, p2):
			distance += np.power(i-j,2)
		distance = np.sqrt(distance)
		return distance

	def fit(self, data, n_neighbors):
		N,D = data.shape 
		labels = [None] * N
		for i, point in enumerate(data):
			distances = list(map(lambda x: self.distance(point, x), self.data))
			n_labels_idx = np.array(distances).argsort()[:n_neighbors]
			n_labels = self.label[n_labels_idx]
			n_labels_count = {v:k for k,v in dict(Counter(n_labels)).items()}
			label = n_labels_count[max(n_labels_count.keys())]
			labels[i] = label 
		return labels

knn = KNN(coordinates,label)
labels_knn = knn.fit(coordinates, 15) # get knn classified label
(len(label)-sum(labels_knn!=label))/len(label) # calculalate correct rate (accuracy)

# classification report 
# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py
print(
    f"Classification report for classifier:\n"
    f"{metrics.classification_report(label, labels_own)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(label, labels_own)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

print(
    f"Classification report for classifier:\n"
    f"{metrics.classification_report(label, labels_module)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(label, labels_module)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
