import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import pickle

trait='Bushy Eyebrows'

with open('filename.pickle', 'rb') as handle:
    Attributes = pickle.load(handle)

# print(Attributes['Jane_Fonda_1'])

X_train,X_test,Y_test,Y_train=[],[],[],[]
Train_Files=os.listdir("./Train_Set")
Test_Files=os.listdir("./Test_Set")

for name in Train_Files:
	x=name.strip('.jpg')
	try:
		Y_train.append(float(Attributes[x][trait]))
		pixels=cv2.imread("./Train_Set/"+name).flatten()
		X_train.append(pixels)
	except:
		pass	

for name in Test_Files:
	x=name.strip('.jpg')
	try:
		Y_test.append(float(Attributes[x][trait]))	
		pixels=cv2.imread("./Test_Set/"+name).flatten()
		X_test.append(pixels)
	except:
		pass	

X_train=np.array(X_train)	
X_test=np.array(X_test)	
Y_train=np.array(Y_train)	
Y_test=np.array(Y_test)	

print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

Y_train[Y_train<=0]=0
Y_train[Y_train>0]=1
Y_test[Y_test<=0]=0
Y_test[Y_test>0]=1

svm = SVC(kernel='rbf')
svm.fit(X_train,Y_train)

Y_pred = svm.predict(X_test)

print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))