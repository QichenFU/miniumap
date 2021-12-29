# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 22:18:17 2021

@author: 17426
"""
# calculate the accuracy
import numpy as np
from collections import Counter 
import pandas as pd
from sklearn.metrics import classification_report

# read umap output (the first 2 columns should be x and y coordinates respectively)
data = pd.read_csv("single_cell_own_output.csv")
coordinates = data.iloc[:,:2].to_numpy()
label = np.array(list(data.iloc[:,2]))
# cahnge label to index to increase speed
label_temp = set(label)
label_temp = list(label_temp)
label_new = []
for i in label:
    label_new.append(label_temp.index(i))
label_new = np.array(label_new)


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

knn = KNN(coordinates,label_new)
labels_knn = knn.fit(coordinates, 15) # get knn classified label
(len(label_new)-sum(labels_knn!=label_new))/len(label_new) # calculalate correct rate (accuracy)
labels_knn_own = labels_knn


from sklearn import metrics
import matplotlib.pyplot as plt
# classification report 
# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py
print(
    f"Classification report for classifier:\n"
    f"{metrics.classification_report(label_new, labels_knn_own)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(label_new, labels_knn_own)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.savefig("own",type="png")
plt.show()
plt.close()

print(
    f"Classification report for classifier:\n"
    f"{metrics.classification_report(label_new, labels_knn_module)}\n"
)

disp = metrics.ConfusionMatrixDisplay.from_predictions(label_new, labels_knn_module)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.savefig("module",type="png")
plt.show()
plt.close()