import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import pickle
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import ShuffleSplit


# with open('filename.pickle', 'rb') as handle:
#     Attributes = pickle.load(handle)

clf = joblib.load('./Att_Classifiers/Asian.pkl')  
print(clf)    
Train_Files=os.listdir("./Train_Set")
Test_Files=os.listdir("./Test_Set")


Chosen=["Asian","Attractive Woman","Baby","Bags Under Eyes","Bald","Teeth Not Visible","Teeth Visible","White","Youth","Sunglasses","Nose-Mouth Lines","Nose Shape","No Beard","Frowning","Blurry","Mustache","No Eyewear","Gray Hair","Eye Width","Smiling"]
# # print(len(Chosen))
for trait in Chosen:
	# trait='Asian'
	X_train,X_test,Y_test,Y_train=[],[],[],[]
	
	for name in Train_Files:
		x=name.strip('.jpg')
		try:
			Y_train.append(float(Attributes[x][trait]))
			pixels=np.load("./Features3/"+x+".npy")
			X_train.append(pixels)
		except:
			pass	

	for name in Test_Files:
		x=name.strip('.jpg')
		try:
			Y_test.append(float(Attributes[x][trait]))	
			pixels=np.load("./Features3/"+x+".npy")
			X_test.append(pixels)
		except:
			pass

	X_train=np.array(X_train)	
	X_test=np.array(X_test)	
	Y_train=np.array(Y_train)	
	Y_test=np.array(Y_test)	

# 	# mean_train=np.median(Y_train)
# 	# mean_test=np.median(Y_test)
# 	mean_train=0
# 	mean_test= 0


# 	# print(len(np.where(Y_train>mean_train)[0]))
	tmp=[]
	for i in range(Y_train.shape[0]):
		if Y_train[i]<mean_train:
			tmp.append(0)
		else:
			tmp.append(1)	
	Y_train=np.array(tmp)
	tmp=[]
	for i in range(Y_test.shape[0]):
		if Y_test[i]<mean_test:
			tmp.append(0)
		else:
			tmp.append(1)	
	Y_test=np.array(tmp)

	print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

# 	# print(len(np.where(Y_train==1)[0]))
# 	# print(len(np.where(Y_train==0)[0]))
# 	# print(len(np.where(Y_test==1)[0]))
# 	# print(len(np.where(Y_test==0)[0]))

	param_grid = {'C': [0.1, 1, 10, 100, 1000],  
	              'gamma': [0.1, 0.01, 0.001, 0.0001], 
	              'kernel': ['rbf']}  
	  
	svm = GridSearchCV(SVC(), param_grid, refit = True, verbose = False) 
	  

	svm.fit(X_train,Y_train)

	fil=trait+".pkl"

	joblib.dump(svm, "./Att_Classifiers/"+fil)

	Y_pred = svm.predict(X_test)

	corr1,corr0,incorr1,incorr0=0,0,0,0
	tp,fp,tn,fn=0,0,0,0
	for i in range(len(Y_pred)):
		if Y_pred[i]==Y_test[i]:
			if Y_test[i]==0:
				corr0+=1
				tn+=1
			else:
				corr1+=1
				tp+=1	
		else:
			if Y_test[i]==0:
				incorr0+=1
				fp+=1
			else:
				fn+=1
				incorr1+=1
	print("#############################################")
	print(trait)
	print(corr0,incorr0)
	print(corr1,incorr1)

	try:		
		recall=(tp/(tp+fn))				
		precision=(tp/(tp+fp))
		f1=(2*(precision*recall))/(precision+recall)
		print("Recall",recall)
		print("Precision",precision)
		print("F1-Score",f1)
	except:
		print("Recall INV")
		print("Precision INV")
		print("F1-Score INV")
								
	print(100*(corr1+corr0)/(corr1+corr0+incorr1+incorr0))
	
	print("#############################################")
	print(confusion_matrix(Y_test,Y_pred))
	print(classification_report(Y_test,Y_pred))

