import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import pickle
from sklearn.externals import joblib

# Load the pretrained model
model = models.resnet18(pretrained=True)# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


def get_vector(image_name):
	img = Image.open(image_name)
	t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
	my_embedding = torch.zeros(1,512,1,1)
	def copy_data(m, i, o):
		my_embedding.copy_(o.data)
	h = layer.register_forward_hook(copy_data)
	model(t_img)
	h.remove()
	return my_embedding

imgname="./Train_Set/Aaron_Eckhart_1.jpg"
img_rep = np.array(get_vector(imgname).flatten())


Files=os.listdir("./Train_Set")
curr=1
for img in Files:
	y="./Features3/"+img.strip(".jpg")
	imgname = "./Train_Set/"+img
	img_rep = np.array(get_vector(imgname).flatten())
	np.save(y,img_rep)
	curr+=1
	a=100*curr/(2*len(Files))
	print("Progress:","%.2f" % a,"%")
curr=1
Files=os.listdir("./Test_Set")
for img in Files:
	y="./Features3/"+img.strip(".jpg")
	imgname = "./Test_Set/"+img
	img_rep = np.array(get_vector(imgname).flatten())
	np.save(y,img_rep)
	curr+=1
	a=100*curr/(2*len(Files))
	print("Progress:","%.2f" % a,"%")    