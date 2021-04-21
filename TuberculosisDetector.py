from sklearn.neighbors import KNeighborsClassifier
import time
import os
from PIL import Image
import numpy as np
import random
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import config as cfg

#Returns true positives, false positives, true negatoves, and false negatives as a percentage using sklearn's confusion matrix
def metrics(true_values, predicted_values):
    cmatrix = confusion_matrix(true_values, predicted_values)
    TP=0
    FP=0
    FN=0
    TN=0

    if len(cmatrix) == 2:
        TP=cmatrix[0][0] / sum(cmatrix[0])
        FP=cmatrix[0][1] / sum(cmatrix[0])
        FN=cmatrix[1][0] / sum(cmatrix[1])
        TN=cmatrix[1][1] / sum(cmatrix[1])
        print("TP: ", cmatrix[0][0] / sum(cmatrix[0]))
        print("FP: ", cmatrix[0][1] / sum(cmatrix[0]))
        print("FN: ", cmatrix[1][0] / sum(cmatrix[1]))
        print("TN: ", cmatrix[1][1] / sum(cmatrix[1]))
    else:
        print(cmatrix)

    return TP,FP,FN,TN

#X-rays are converted into feature vectors
#Dataset has been manually split up into healthy and tuberculosis images beforehand
normal_features = []
normal_labels = []
for img in os.listdir(cfg.healthy):
	if str(img).endswith('.png'):
		label = 'normal'
		img = Image.open(cfg.healthy + str(img))
		img = img.convert('RGB')
		img = img.resize((300, 300), Image.ANTIALIAS)
		normal_features.append(np.array(img))
		normal_labels.append(label)
random.shuffle(normal_features)

tuberculosis_features = []
tuberculosis_labels = []
for img in os.listdir(cfg.tuberculosis):
	if str(img).endswith('.png'):
		label = 'tuberculosis'
		img = Image.open(cfg.tuberculosis + str(img))
		img = img.convert('RGB')
		img = img.resize((300, 300), Image.ANTIALIAS)
		tuberculosis_features.append(np.array(img))
		tuberculosis_labels.append(label)
random.shuffle(tuberculosis_features)

#Dataset is divided 70% into training, 30% into testing
train_features = np.array(normal_features[:228] + tuberculosis_features[:228])
train_labels = np.array(normal_labels[:228] + tuberculosis_labels[:228])
test_features = np.array(normal_features[229:328] + tuberculosis_features[229:328])
test_labels = np.array(normal_labels[229:328] + tuberculosis_labels[229:328])

#Dataset is reshaped so it can be used with k-NN
train_features = np.reshape(train_features, (train_features.shape[0], -1))
test_features = np.reshape(test_features, (test_features.shape[0], -1))

#Classifier is trained
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=10, algorithm='brute', weights='distance')
knn.fit(train_features, train_labels)
print("Training time of KNN: --- %s seconds ---" % (time.time() - start_time))

#Predictions on training data
start_time = time.time()
predicted_train = knn.predict(train_features)
print("Training data prediction time: --- %s seconds ---" % (time.time() - start_time))
print(classification_report(train_labels, predicted_train))
train_TP,train_FP,train_FN,train_TN=metrics(train_labels, predicted_train)

#Predictions on testing data
start_time = time.time()
predicted_test = knn.predict(test_features)
print("\nTesting data prediction time: --- %s seconds ---" % (time.time() - start_time))
print(classification_report(test_labels, predicted_test))
test_TP,test_FP,test_FN,test_TN=metrics(test_labels, predicted_test)


