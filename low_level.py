import numpy as np 
import cv2
import math
from matplotlib import pyplot as plt

cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

img = "./Train_Set/Ahmet_Demir_1.jpg"

def expand_bb(x, y, w, h, img):
	
	#factor by which to extend in all corners
	factor = 0.1


	tlx, tly, brx, bry = x, y, x+w, y+h
	if tlx - math.floor(factor*w) >= 0:
		tlx -= math.floor(factor*w)
	else:
		tlx = 0
	if tly - math.floor(factor*h) >= 0:
		tly -= math.floor(factor*h)
	else:
		tly = 0
	if brx + math.floor(factor*w) <= img.shape[0]:
		brx += math.floor(factor*w)
	else:
		brx = img.shape[0]
	if bry + math.floor(factor*h) <= img.shape[1]:
		bry += math.floor(factor*h)
	else:
		bry = img.shape[1]
	
	return tlx, tly, brx-tlx, bry-tly

def low_level(img):
	#extracting faces
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,flags=cv2.CASCADE_SCALE_IMAGE)

	#finding the largest face
	sz = faces[:, 2] * faces[:, 3]
	face_ind = np.argmax(sz)
	x, y, w, h = faces[face_ind, :]

	#expanding the bounding box
	x, y, w, h = expand_bb(x, y, w, h, img)
	# print(x, y, w ,h)
	# plt.imshow(img[x:x+w, y:y+h, :])
	# plt.show()

	return img[x:x+w, y:y+h, :]

low_level(cv2.imread(img))

	

