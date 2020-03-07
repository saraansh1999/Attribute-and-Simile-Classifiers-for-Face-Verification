import numpy as np 
import cv2
import math
import dlib
from imutils import face_utils
from matplotlib import pyplot as plt

cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
landmarkPath = "./shape_predictor_68_face_landmarks.dat"
landmarkPredictor = dlib.shape_predictor(landmarkPath)

def normalize(img):
	# for i in range(img.shape[2]):
	# 	img[:, :, i] = (img[:, :, i] - np.mean(img[:, :, i])) 
	# return np.round(((img - np.min(img)) / (np.max(img) - np.min(img))) * 255)
	return (img - np.mean(img)) / np.std(img)
	# return (img - np.mean(img))
	return img

def expand_bb(x, y, w, h, img, factor):

	tlx, tly, brx, bry = x, y, x+w, y+h
	if tlx - math.floor(factor*w) >= 0:
		tlx -= math.floor(factor*w)
	else:
		tlx = 0
	if tly - math.floor(factor*h) >= 0:
		tly -= math.floor(factor*h)
	else:
		tly = 0
	if brx + math.floor(factor*w) <= img.shape[1]:
		brx += math.floor(factor*w)
	else:
		brx = img.shape[0]
	if bry + math.floor(factor*h) <= img.shape[0]:
		bry += math.floor(factor*h)
	else:
		bry = img.shape[1]
	
	return tlx, tly, brx-tlx, bry-tly


def low_level(img):
	
	#extracting faces
	gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)

	#finding the largest face
	sz = faces[:, 2] * faces[:, 3]
	face_ind = np.argmax(sz)
	x, y, w, h = faces[face_ind, :]

	#expanding the bounding box
	x, y, w, h = expand_bb(x, y, w, h, img, 0.1)

	faceimg = img[y:y+h, x:x+w, :]
	print(np.max(faceimg), np.min(faceimg))

	#predicting facial landmarks
	shape = landmarkPredictor(faceimg, dlib.rectangle(0, 0, w, h))
	shape = face_utils.shape_to_np(shape)

	#extracting regions coordinates
	regions = {}
	eyes = []
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

		#create ROIs
		rx, ry, rw, rh = cv2.boundingRect(np.array([shape[i:j]]))
		rx, ry, rw, rh = expand_bb(rx, ry, rw, rh, faceimg, 0.3)
		regions[name] = [rx, ry, rw, rh]
		#combining eyes and eyebrows
		if name.split("_")[0] == "left" or name.split("_")[0] == "right":
			eyes.extend(shape[i:j])
	eyes = np.array(eyes).squeeze()
	rx, ry, rw, rh = cv2.boundingRect(np.array(eyes))
	rx, ry, rw, rh = expand_bb(rx, ry, rw, rh, faceimg, 0.3)
	regions["eyes"] = [rx, ry, rw, rh]
	regions["full"] = [0, 0, faceimg.shape[1], faceimg.shape[0]]

	#extracting region wise features
	features = {}
	for key in regions:
		rx, ry, rw, rh = regions[key]
		regions[key] = faceimg[ry:ry+rh, rx:rx+rw, :]
		
		#RGB features
		features[key+"_RGB"], _ = np.histogram(normalize(regions[key]), bins=256)

		#HSV features
		features[key+"_HSV"], _ = np.histogram(normalize(cv2.cvtColor(regions[key], cv2.COLOR_BGR2HSV)), bins=256)

		#gradient features
		gx = cv2.Sobel(regions[key], cv2.CV_64F, 1, 0, ksize=5)
		gy = cv2.Sobel(regions[key], cv2.CV_64F, 1, 0, ksize=5)
		features[key+"_gradientMagnitude"], _ = np.histogram(normalize(np.sqrt(np.square(gx) + np.square(gy))), bins=256)
		temp = np.arctan2(gy, gx)
		features[key+"_gradientOrientation"], _ = [np.mean(temp), np.std(temp)]
		
		#visualising everything
		fig, ax = plt.subplots(3, 2)
		ax[0, 0].imshow(regions[key][:, :, ::-1])
		ax[0, 0].title.set_text(key)
		ax[0, 1].hist(features[key+"_RGB"], bins=np.arange(256))
		ax[0, 1].title.set_text(key+"_RGB")
		ax[1, 0].hist(features[key+"_HSV"], bins=np.arange(256))
		ax[1, 0].title.set_text(key+"_HSV")
		ax[1, 1].hist(features[key+"_gradientMagnitude"], bins=np.arange(256))
		ax[1, 1].title.set_text(key+"_gradientMagnitude")
		ax[2, 1].imshow(normalize(np.sqrt(np.square(gx) + np.square(gy))))
		ax[2, 1].title.set_text("Gradient image")
		plt.show()


	return None

img = "Train_Set/Matt_LeBlanc_1.jpg"
low_level(cv2.imread(img))