import numpy as np
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

List=os.listdir("./dir_009")
Feats={}
ref_people=[]
for i in List:	
	newname=i.replace(' ','')
	ref_people.append(i)
	# P=os.listdir("./Simile_feat/"+i)
	# for k in P:
	# 	Feats[newname].append(np.load("./Simile_feat/"+i+"/"+k,allow_pickle='TRUE').item())

region=["eyes_RGB","nose_RGB","right_eyebrow_RGB","left_eyebrow_RGB","mouth_RGB"]


		
i="ReneeZellweger"
j="eyes_RGB"

classifier_name=i+"_"+j

X1,Y1=[],[]
X2,Y2=[],[]
X3,Y3=[],[]
L=os.listdir("./Simile_Feat/ReneeZellweger")
for f in L:
	feat=np.load("./Simile_Feat/ReneeZellweger/"+f,allow_pickle='TRUE').item()
	X1.append(feat[j])
	Y1.append(1)

L=np.array(os.listdir("./train_feat/"))
L=L[np.random.choice(len(L),min(len(L),800))]
for fol in L:
	P=np.array(os.listdir("./train_feat/"+fol))
	P=P[np.random.choice(len(P),min(len(P),1))]
	for fil in P:
		feat=np.load("./train_feat/"+fol+"/"+fil,allow_pickle='TRUE')
		X2.append(feat[j])
		Y2.append(0)

L=np.array(os.listdir("./test_feat/"))
L=L[np.random.choice(len(L),min(len(L),200))]
for fol in L:
	P=np.array(os.listdir("./test_feat/"+fol))
	P=P[np.random.choice(len(P),min(len(P),1))]
	for fil in P:
		feat=np.load("./test_feat/"+fol+"/"+fil,allow_pickle='TRUE')
		X3.append(feat[j])
		Y3.append(0)

X1,Y1=np.array(X1),np.array(Y1)
X2,Y2=np.array(X2),np.array(Y2)
X3,Y3=np.array(X3),np.array(Y3)
PP,PP1=[],[]

idx=np.random.choice(X1.shape[0],X1.shape[0]-200)
for i in range(X1.shape[0]):
	if i not in idx:
		PP.append(X1[i]),PP1.append(Y1[i])
		
PP,PP1=np.array(PP),np.array(PP1)
X_train=np.concatenate([X2,X1[idx]])
Y_train=np.concatenate([Y2,Y1[idx]])
X_test=np.concatenate([X3,PP])
Y_test=np.concatenate([Y3,PP1])


# print("wegggggggggggggggggggggg")
print(PP.shape,X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Training
######################################################
print("Train Data Shape -> ", X_train.shape)
print("Positive samples -> ", Y_train[Y_train == 1].shape)
print("Negative samples -> ", Y_train[Y_train == 0].shape)

param_grid = {'C': [5, 10, 20, 30,15],  
              'gamma': ['scale'], 
              'kernel': ['rbf']}  
  
svm = GridSearchCV(SVC(probability=True), param_grid, scoring='f1', refit = True, verbose = 3, cv=2) 

svm.fit(X_train,Y_train)

print("Best params: ", svm.best_estimator_)
print("Train " + "metric" + ": ", svm.best_score_)
######################################################

# saving model
######################################################
with open("./Simile_Classifiers/" + classifier_name + ".pkl", 'wb') as f:
	pickle.dump(svm, f)
######################################################

# Testing
######################################################
print("Test Data Shape -> ", X_test.shape)
Y_pred = svm.predict(X_test)
print("Confusion Matrix: \n", confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
######################################################

