
#import sys
#import os
#import json
#import numpy

from sklearn import neighbors#, datasets
#from dtw import dtw

#from sklearn import svm




#X = dataset
X = [[2,2],[1,2],[3,4],[5,4]]
#y = numpy.array(class_label)#Labels for classes
y = ["asdasd",1,1,4]
knn = neighbors.KNeighborsClassifier(n_neighbors = 1)
knn.fit(X,y)
print knn.predict([[2,2]])
#print knn.predict_proba([[1,2,1,1,1,1]])
