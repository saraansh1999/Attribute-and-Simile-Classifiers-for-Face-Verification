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
import random

Train_Pairs=[]
Test_Pairs=[]

images=np.array(os.listdir("./train"))

Rand=images[np.random.choice(images.shape[0],4000)]
for i in range(0,4000,2):
	A=os.listdir("./train/"+Rand[i])
	B=os.listdir("./train/"+Rand[i+1])

	f1,f2=Rand[i]+"/"+A[random.randint(0,len(A)-1)],Rand[i+1]+"/"+B[random.randint(0,len(B)-1)]

	Train_Pairs.append([f1,f2,0])

for img_fol in images:

	fils=np.array(os.listdir("./train/"+img_fol))
	if fils.shape[0]==2:
		Train_Pairs.append([img_fol+"/"+fils[0],img_fol+"/"+fils[1],1])
	if fils.shape[0]>2:
		t=fils[np.random.choice(len(fils),4)]
		Train_Pairs.append([img_fol+"/"+t[0],img_fol+"/"+t[1],1])	
		Train_Pairs.append([img_fol+"/"+t[2],img_fol+"/"+t[3],1])

images=np.array(os.listdir("./test"))
Rand=images[np.random.choice(images.shape[0],1000)]

for i in range(0,1000,2):
	A=os.listdir("./test/"+Rand[i])
	B=os.listdir("./test/"+Rand[i+1])

	f1,f2=Rand[i]+"/"+A[random.randint(0,len(A)-1)],Rand[i+1]+"/"+B[random.randint(0,len(B)-1)]

	Train_Pairs.append([f1,f2,0])
for img_fol in images:
	fils=np.array(os.listdir("./test/"+(img_fol)))
	if fils.shape[0]==2:
		Test_Pairs.append([img_fol+"/"+fils[0],img_fol+"/"+fils[1],1])
	if fils.shape[0]>2:
		t=fils[np.random.choice(len(fils),4)]
		Test_Pairs.append([img_fol+"/"+t[0],img_fol+"/"+t[1],1])	
		Test_Pairs.append([img_fol+"/"+t[2],img_fol+"/"+t[3],1])


# Train_Pairs=np.array(Train_Pairs)
# Test_Pairs=np.array(Test_Pairs)
# print(Train_Pairs.shape,Test_Pairs.shape)
# print(Train_Pairs[0],Test_Pairs[0])

def op(X1,X2):
	return np.concatenate[np.abs(X1-X2),np.sqrt(X1*X2),(X1+X2)/2]


X_Train,Y_Train,Y_Test,X_Test=[],[],[],[]

for i in Train_Pairs:
	Y_Train.append(int(i[2]))
	f1=np.load("./Final_Features/"+i[0],allow_pickle='TRUE').item()	
	f2=np.load("./Final_Features/"+i[1],allow_pickle='TRUE').item()	
	X_Train.append(op(f1,f2))

for i in Test_Pairs:
	Y_Test.append(int(i[2]))
	f1=np.load("./Final_Features/"+i[0],allow_pickle='TRUE').item()	
	f2=np.load("./Final_Features/"+i[1],allow_pickle='TRUE').item()	
	X_Test.append(op(f1,f2))

X_Train=np.array(X_Train)
X_Test=np.array(X_Test)
Y_Train=np.array(Y_Train)
Y_Test=np.array(Y_Test)