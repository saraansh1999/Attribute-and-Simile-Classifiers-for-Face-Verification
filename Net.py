import numpy as np
import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report



Train_Files=os.listdir("./Train_Set")
Test_Files=os.listdir("./Test_Set")

with open('filename.pickle', 'rb') as handle:
    Attributes = pickle.load(handle)

trait=sys.argv[1]

X_train,X_test,Y_test,Y_train=[],[],[],[]

for name in Train_Files:
	x=name.strip('.jpg')
	try:
		# print(x)
		Y_train.append(float(Attributes[x][trait]))
		pixels=np.load("./Features2/"+x+".npy")
		X_train.append(pixels)
	except:
		pass	

for name in Test_Files:
	x=name.strip('.jpg')
	try:
		Y_test.append(float(Attributes[x][trait]))	
		pixels=np.load("./Features2/"+x+".npy")
		X_test.append(pixels)
	except:
		pass

X_train=np.array(X_train)	
X_test=np.array(X_test)	
Y_train=np.array(Y_train)	
Y_test=np.array(Y_test)	

Y_train[Y_train<=0]=0
Y_train[Y_train>0]=1
Y_test[Y_test<=0]=0
Y_test[Y_test>0]=1

print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

EPOCHS = 10
LEARNING_RATE = 0.001

class trainData(Dataset):
    
	def __init__(self, X_data, y_data):
	    self.X_data = X_data
	    self.y_data = y_data
	    
	def __getitem__(self, index):
	    return self.X_data[index], self.y_data[index]
	    
	def __len__ (self):
	    return len(self.X_data)


train_data = trainData(torch.FloatTensor(X_train),torch.FloatTensor(Y_train))

class testData(Dataset):   
	def __init__(self, X_data):
	    self.X_data = X_data
	    
	def __getitem__(self, index):
	    return self.X_data[index]
	    
	def __len__ (self):
	    return len(self.X_data)
    

test_data = testData(torch.FloatTensor(X_test))



class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()        
        self.layer_1 = nn.Linear(512, 128) 
        self.layer_2 = nn.Linear(128, 32)
        self.layer_out = nn.Linear(32, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x


model = DeepNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def Acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    lol=y_pred_tag.cpu().detach().numpy()
    acc=np.where(lol==y_test.reshape(4012,1))[0].shape[0]
    return 100*(acc/lol.shape[0])

model.train()

Y_train1=Y_train
for epo in range(EPOCHS):
	
	epoch_loss = 0
	epoch_acc = 0
	
	optimizer.zero_grad()
	
	y_pred = model(torch.FloatTensor(X_train))
	

	loss = criterion(y_pred, torch.FloatTensor(Y_train.reshape(4012,1)))
	acc = Acc(y_pred, Y_train1)
	
	loss.backward()
	optimizer.step()

	epoch_loss += loss.item()
	epoch_acc += acc
	
	print("Epoch:",epo)
	print("Loss:",epoch_loss)
	print("Acc:",epoch_acc)
	print("##################################################")


model.eval()

y_test_pred = model(torch.FloatTensor(X_test))
y_test_pred = torch.sigmoid(y_test_pred)
y_pred_tag = torch.round(y_test_pred)


confusion_matrix(Y_test, y_pred_tag.cpu().detach().numpy())
print(classification_report(Y_test, y_pred_tag.cpu().detach().numpy()))

# torch.save(model.state_dict(), "./Nets_Attributes/"+trait)

# # model.load_state_dict(torch.load(filepath))
# # model.eval()