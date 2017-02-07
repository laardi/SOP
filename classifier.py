import os
from numpy import array
from math import isnan

from sklearn import neighbors
from sklearn import svm
#from dtw import dtw

#from sklearn.svm import SVC

#import matplotlib.pyplot
#from matplotlib.colors import ListedColormap

class Classifier():
    def __init__(self, database):
        self.db = database
        self.used_features = []
        self.result = -1
        self.result_name = ""
        self._result = ""
        #self.max_samples = 0       #not implemented
        self.VERBOSE = False
        self.ignored_file = ""
    def create_class_structure(self, audio_class, samples):
        try:
            class_directory = os.path.join("classes", "natural")
            os.makedirs(class_directory)
            class_directory = os.path.join("classes", "non-natural")
            os.makedirs(class_directory)
            class_directory = os.path.join("classes", "speech")
            os.makedirs(class_directory)
            class_directory = os.path.join("classes", "music")
            os.makedirs(class_directory)
        except OSError:
            pass

        try:
            class_directory = os.path.join("classes", audio_class)
            os.makedirs(class_directory)
        except OSError:
            pass

        # new_audio = Audio()
        # new_audio.pre_process(samples)

    def delete_class(self):
        pass

    def classify(self, audio_file, timestamp=None, algorithm="knn", neighbors_knn=5):
        data = self.db
        self.ignored_file = audio_file.filename
        return self._classify(data, audio_file, algorithm, neighbors_knn)

    # recursive classifier
    def _classify(self, data, audio_file, algorithm="knn", neighbors_knn=3):
        samples = []
        class_ids = []
        class_names = []
        n_classes = 0
        i = 0

        for node in data:
            if (type(data[node]) == dict) and ('samples' in data[node]):
                self.get_samples(data[node], samples, class_ids, n_classes)
                class_names.append(node)
                n_classes += 1

        if n_classes > 0:
            if self.VERBOSE:
                print '-----------'
                print 'Classifying:'
                print 'nr of classes: ', n_classes
                print 'Classes:', class_names
                print 'nr of samples: ', len(samples)

            X, y, Z = self.create_datasets(samples, class_ids, audio_file.features)

            if algorithm == "knn":
                self.result = self._classify_knn(X, y, Z, neighbors_knn)

            elif algorithm == "svm":
                self.result = self._classify_svm(X, y, Z)

            self.result_name = class_names[self.result]
            if self.VERBOSE:
                print 'Result: ', self.result_name
                print '--------------------------------------------'
            return self._classify(data[self.result_name], audio_file, algorithm)

        else:
            return self.result_name

    # get sound samples recursively
    def get_samples(self, data, samples, class_ids, id):
        for node in data:
            if node == 'samples':
                for i in range(0, len(data['samples'])):
                    if self.ignored_file != data['filenames'][i]:
                        samples.append(data['samples'][i])
                        class_ids.append(id)
                        #print 'sample added'
                    else:
                        if self.VERBOSE:
                            print 'sample ignored:', data['filenames'][i]
                #if (self.VERBOSE) and (len(data[node]) > 0):    #-1
                #    print 'get_samples(): Got %d samples.' % len(data[node])
            if (type(data[node]) == dict) and ('samples' in data[node]):
                if self.VERBOSE:
                    print 'get_samples(): go to subclass: ', node
                self.get_samples(data[node], samples, class_ids, id)

    # create datasets X, y, Z = samples, class_labels, audio_features
    def create_datasets(self, samples, class_ids, features):
        class_features = []
        audio_features = []
        n_features = 0
        n_classes = 0
        n_samples = 0
        audio_f_added = False
        for sample in samples:
            for feature, value in features.items():
                #print "feature: ", feature, value
                if feature in sample and [i for i in self.used_features if i in feature]:##feature in self.used_features:
                    if type(value) == list: # e.g. mfcc
                        for i,j in enumerate(sample[feature]):
                            if isnan(j):
                                sample[feature][i] = 0

                        class_features += sample[feature]
                        if not audio_f_added:   # do only once
                            audio_features += value
                            n_features += len(value)
                    else:
                        class_features.append(sample[feature])
                        if not audio_f_added:
                            audio_features.append(value)
                            n_features += 1
            audio_f_added = True
            n_samples += 1
        training_data = array(class_features)
        X = training_data.reshape(n_samples, n_features)
        y = array(class_ids)
        Z = array(audio_features)
        return X, y, Z

    def _classify_knn(self, X, y, Z, n):
        knn = neighbors.KNeighborsClassifier(n_neighbors = int(n))
        knn.fit(X, y)
        result_knn = knn.predict(Z)
        return int(result_knn)

    def _classify_svm(self, X, y, Z):
        svm_clf = svm.SVC()
        svm_clf.fit(X, y)
        result_svm = svm_clf.predict(Z)
        return int(result_svm)
