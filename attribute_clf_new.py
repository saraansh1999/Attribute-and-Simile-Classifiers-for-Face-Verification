import os
import pickle
import cv2
import numpy as np
import argparse
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.metrics import *

def get_name(file):
	name = file.split('.')[-2]
	id = int(name.split('_')[-1])
	name = '_'.join(name.split('_')[:-1] + [str(id)])
	return name

def check_unbalance(x, y, thresh):
	ones = np.where(y == 1)[0]
	zeros = np.where(y == 0)[0]
	if ones.shape[0] / zeros.shape[0] < thresh:
		zeros = np.random.permutation(zeros)[:np.int(ones.shape[0] / thresh)]
	elif zeros.shape[0] / ones.shape[0] < thresh:
		ones = np.random.permutation(ones)[:np.int(zeros.shape[0] / thresh)]
	newx = np.vstack((x[ones], x[zeros]))
	newy = np.hstack((y[ones], y[zeros]))
	return newx, newy

parser = argparse.ArgumentParser()
parser.add_argument('--trait', action='store', required=True)
parser.add_argument('--metric', action='store', default='f1')
args = parser.parse_args()

# global variables
######################################################
trait = args.trait  
print("Trait -> ", trait)
metric = args.metric
sample_thresh = 1
######################################################

# loading attributes file
######################################################
with open('attr.pkl', 'rb') as handle:
    Attributes = pickle.load(handle)
######################################################

# loading data
######################################################
X_train, X_test, Y_test, Y_train = [], [], [], []
train_dir = "./train_feat"
test_dir = "./test_feat"

for root, dirs, files in os.walk(train_dir):
	for file in files:
		fpath = os.path.join(root, file)
		name = get_name(file)
		try:
			gt = float(Attributes[name][trait]) > 0
			Y_train.append(gt)
			features = np.load(fpath, allow_pickle=True)
			inp = []
			for k in features:
				inp.extend(features[k])
			X_train.append(inp)
		except:
			pass
			# print("Data missing for ", name)
for root, dirs, files in os.walk(test_dir):
	for file in files:
		fpath = os.path.join(root, file)
		name = get_name(file)
		try:
			gt = float(Attributes[name][trait]) > 0
			Y_test.append(gt)
			features = np.load(fpath, allow_pickle=True)
			inp = []
			for k in features:
				inp.extend(features[k])
			X_test.append(inp)
		except:
			pass
			# print("Data missing for ", name)

X_train, X_test = np.array(X_train), np.array(X_test)
Y_train, Y_test = np.array(Y_train), np.array(Y_test)

X_train, Y_train = check_unbalance(X_train, Y_train, sample_thresh)
######################################################

# feature selection
######################################################
feat_sel = SelectKBest(f_classif, k=500)
X_train = feat_sel.fit_transform(X_train, Y_train)
######################################################

# Training
######################################################
print("Train Data Shape -> ", X_train.shape)
print("Positive samples -> ", Y_train[Y_train == 1].shape)
print("Negative samples -> ", Y_train[Y_train == 0].shape)

param_grid = {'C': [10, 20, 30],  
              'gamma': ['scale'], 
              'kernel': ['poly']}  
  
svm = GridSearchCV(SVC(probability=True), param_grid, scoring=metric, refit = True, verbose = 3, cv=2) 

svm.fit(X_train,Y_train)

print("Best params: ", svm.best_estimator_)
print("Train " + metric + ": ", svm.best_score_)
######################################################

# saving model
######################################################
with open("./Att_Classifiers/" + trait + ".pkl", 'wb') as f:
	pickle.dump(svm, f)
######################################################

# Testing
######################################################
print("Test Data Shape -> ", X_test.shape)
Y_pred = svm.predict(feat_sel.transform(X_test))
print("Confusion Matrix: \n", confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
######################################################
