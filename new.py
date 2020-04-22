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

try:
	os.system("mkdir ./Final_Features")
except:
	 pass	

CL_A,CL_S=[],[]
SLC=os.listdir("./Simile_Classifiers")
ATC=os.listdir("./Att_Classifiers")

for i in SLC:
	with open('./Simile_Classifiers/'+i, 'rb') as f:
    	CL_S.append(pickle.load(f),)

for i in ATC:
	with open('./Att_Classifiers/'+i, 'rb') as f:
    	CL_A.append(pickle.load(f))

P=os.listdir("./train_feat")
for img in P:
	K=os.listdir("./train_feat"+img)
	for subimg in K:
		feat=np.load("./train_feat/"+img+"/"+subimg,allow_pickle='TRUE')

		Rep=np.array([])
		for key in feat:
			if "eyes" in key and "img" not in key:
				Rep=np.concatenate([Rep,feat[key]])
		x=0
		Rep1=np.array([])
		Rep2=np.array([])
		Rep3=np.array([])
		for key in feat:	
			if "img" not in key and key is not "full_RGB":
				if x<5 and x>=0:
					Rep1=np.concatenate([Rep1,feat[key]])
				if x<10 and x>=5:
					Rep2=np.concatenate([Rep2,feat[key]])
				if x<15 and x>=10:
					Rep3=np.concatenate([Rep3,feat[key]])
				x+=1

		i,Feature_Rep=[],[]
		for i in CL_S:
			Feature_Rep.append(i.predict_proba([Rep])[0][0])
		for i in CL_A:
			Feature_Rep.append(i.predict_proba([Rep1])[0][0])
			Feature_Rep.append(i.predict_proba([Rep2])[0][0])
			Feature_Rep.append(i.predict_proba([Rep3])[0][0])	

		np.save("./Final_Features/"+img+"/"+subimg.strip('.jpg'),Feature_Rep)
	
		


