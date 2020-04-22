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
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

def op(X1,X2):
	return np.concatenate([np.abs(X1-X2),np.sqrt(X1*X2),(X1+X2)/2])

with open('./Final_Verification_Classifier.pkl', 'rb') as f:
    	clf=pickle.load(f)

base="./Final_Features/"

f1 = askopenfilename()
f2 = askopenfilename()

dir1,fol1,fil1=f1.split('/')[-3],f1.split('/')[-2],f1.split('/')[-1]
dir2,fol2,fil2=f2.split('/')[-3],f2.split('/')[-2],f2.split('/')[-1]

f1=base+fol1+"/"+fil1
f2=base+fol2+"/"+fil2

img1name=dir1+"/"+fol1+"/"+fil1
img2name=dir2+"/"+fol2+"/"+fil2



f1=f1.replace('jpg','npy')
f2=f2.replace('jpg','npy')


feat1=np.load(f1,allow_pickle='TRUE')	
feat2=np.load(f2,allow_pickle='TRUE')



print(clf.predict([op(feat1,feat2)]))
	
I1 = cv2.cvtColor(cv2.imread(img1name), cv2.COLOR_BGR2RGB)
I2 = cv2.cvtColor(cv2.imread(img2name), cv2.COLOR_BGR2RGB)

# print(img1name)

plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.imshow(I1)
plt.subplot(1,2,2)
plt.imshow(I2)
plt.show()
	
		

