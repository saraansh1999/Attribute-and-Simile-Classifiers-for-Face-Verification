import numpy as np 
import cv2
import math
import dlib
import copy
from imutils import face_utils
from matplotlib import pyplot as plt

cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
landmarkPath = "./shape_predictor_68_face_landmarks.dat"
landmarkPredictor = dlib.shape_predictor(landmarkPath)
VIS = True
SZ = 50

def normalize(imgorig, method="standardization"):
	img = copy.deepcopy(imgorig)
	if method == "channelwise_mean_only":
		# for i in range(img.shape[2]):
		# 	print(np.min(img[:, :, i]), np.mean(img[:, :, i]), np.max(img[:, :, i]))
		img = img - np.mean(img, axis=(0, 1))
			# img[:, :, i] = (img[:, :, i] - np.mean(img[:, :, i])) 
			# print(np.min(img[:, :, i]), np.mean(img[:, :, i]), np.max(img[:, :, i]))
		return img
	if method == "channelwise_standardization":
		for i in range(img.shape[2]):
			img[:, :, i] = (img[:, :, i] - np.mean(img[:, :, i])) / np.std(img[:, :, i]) 
		return img
	if method == "channelwise_max_min":
		img = np.round((img - np.min(img, axis=(0,1))) / (np.max(img, axis=(0,1)) - np.min(img, axis=(0,1))) * 255)
		# for i in range(img.shape[2]):
		# 	img[:, :, i] = np.round(((img[:, :, i] - np.min(img[:, :, i])) /  (np.max(img[:, :, i]) - np.min(img[:, :, i]))) *255)
		return img
	if method == "max_min":
		return np.round(((img - np.min(img)) / (np.max(img) - np.min(img))) * 255)
	if method == "standardization":
		return (img - np.mean(img)) / np.std(img)
	if method == "mean_only":
		return (img - np.mean(img))

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

def get_face_coords(img):
	#extracting faces
	gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)

	#finding the largest face
	sz = faces[:, 2] * faces[:, 3]
	face_ind = np.argmax(sz)
	x, y, w, h = faces[face_ind, :]

	#expanding the bounding box
	return expand_bb(x, y, w, h, img, 0.2)	

def findhist(img, bins):
	# hist = []
	# for i in range(img.shape[2]):
	# 	a, b = np.histogram(img[:, :, i].flatten(), bins = bins)
	# 	hist.append(a)
	# hist = np.reshape(np.array(hist[:]), -1)
	# return hist
	a, b = np.histogram(img.flatten(), bins = bins)
	return a

def low_level(img):
	
	#extracting faces
	gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
	x, y, w, h = get_face_coords(img)
	unrotated_face = img[y:y+h, x:x+w]

	#predicting facial landmarks
	shape = landmarkPredictor(gray, dlib.rectangle(x, y, x+w, y+h))
	shape = face_utils.shape_to_np(shape)

	# Affine transformation
	# extract the left and right eye (x, y)-coordinates
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	leftEyePts = shape[lStart:lEnd]
	rightEyePts = shape[rStart:rEnd]
	# compute the center of mass for each eye
	leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
	rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
	# compute the angle between the eye centroids
	dY = rightEyeCenter[1] - leftEyeCenter[1]
	dX = rightEyeCenter[0] - leftEyeCenter[0]
	angle = np.degrees(np.arctan2(dY, dX)) - 180
	desiredLeftEye=(0.35, 0.35)
	desiredFaceWidth=200
	desiredFaceHeight=200
	# compute the desired right eye x-coordinate based on the
	# desired x-coordinate of the left eye
	desiredRightEyeX = 1.0 - desiredLeftEye[0]
	# determine the scale of the new resulting image by taking
	# the ratio of the distance between eyes in the *current*
	# image to the ratio of distance between eyes in the
	# *desired* image
	dist = np.sqrt((dX ** 2) + (dY ** 2))
	desiredDist = (desiredRightEyeX - desiredLeftEye[0])
	desiredDist *= desiredFaceWidth
	scale = desiredDist / dist
	# compute center (x, y)-coordinates (i.e., the median point)
	# between the two eyes in the input image
	eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
		(leftEyeCenter[1] + rightEyeCenter[1]) // 2)
	# grab the rotation matrix for rotating and scaling the face
	M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
	# update the translation component of the matrix
	tX = desiredFaceWidth * 0.5
	tY = desiredFaceHeight * desiredLeftEye[1]
	M[0, 2] += (tX - eyesCenter[0])
	M[1, 2] += (tY - eyesCenter[1])
	# apply the affine transformation
	(w, h) = (desiredFaceWidth, desiredFaceHeight)
	faceimg = cv2.warpAffine(img, M, (w, h),
		flags=cv2.INTER_CUBIC)

	if VIS:
		fig, ax = plt.subplots(1, 2)
		ax[0].imshow(unrotated_face[:, :, ::-1])
		ax[0].title.set_text("Original")
		ax[1].imshow(faceimg[:, :, ::-1])
		ax[1].title.set_text("Affine Tranformed")
		plt.show()

	#predicting facial landmarks again in the transformed image
	shape = landmarkPredictor(faceimg, dlib.rectangle(0, 0, w, h))
	shape = face_utils.shape_to_np(shape)

	#extracting regions coordinates
	regions = {}
	eyes = []
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

		#create ROIs
		if name == "jaw" or name == "inner_mouth":
			continue
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
	del(regions["left_eye"], regions["right_eye"], regions["left_eyebrow"], regions["right_eyebrow"])


	#extracting region wise features
	features = {}
	for key in regions:
		rx, ry, rw, rh = regions[key]
		regions[key] = faceimg[ry:ry+rh, rx:rx+rw, :]
		
		r = np.arange(0, 256)

		#RGB features
		# print(regions[key])
		features[key+"_RGB"] = findhist(normalize(regions[key], 'channelwise_max_min'), r)

		# # HSV features
		features[key+"_HSV"] = findhist(normalize(cv2.cvtColor(regions[key], cv2.COLOR_BGR2HSV), 'channelwise_max_min'), r)

		# #gradient features
		gray = cv2.cvtColor(regions[key], cv2.COLOR_BGR2GRAY)
		gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
		gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
		gradMag = normalize(np.sqrt(np.square(gx) + np.square(gy)), "max_min")
		features[key+"_gradientMagnitude"] = findhist(gradMag, r)		
		gradOri = normalize(np.arctan2(gy, gx), "max_min")
		# features[key+"_gradientOrientation"], _ = [np.mean(gradOri), np.std(gradOri)]
		features[key+"_gradientOrientation"] = findhist(gradOri, r)
		
		#visualising everything
		if(VIS):
			fig, ax = plt.subplots(3, 2)
			ax[0, 0].imshow(regions[key][:, :, ::-1])
			ax[0, 0].title.set_text(key)
			ax[0, 1].hist(r[:-1], weights=features[key+"_RGB"], bins=r)
			ax[0, 1].title.set_text(key+"_RGB")
			ax[1, 0].hist(r[:-1], weights=features[key+"_HSV"], bins=r)
			ax[1, 0].title.set_text(key+"_HSV")
			ax[1, 1].hist(r[:-1], weights=features[key+"_gradientMagnitude"], bins=r)
			ax[1, 1].title.set_text(key+"_gradientMagnitude")
			ax[2, 0].hist(r[:-1], weights=features[key+"_gradientOrientation"], bins=r)
			ax[2, 0].title.set_text(key+"_gradientOrientation")
			ax[2, 1].imshow(gradMag)
			ax[2, 1].title.set_text("Gradient image")
			plt.show()


	features["gradimg"] = cv2.resize(gradMag, (SZ, SZ)).flatten()
	features["grayimg"] = cv2.resize(gray, (SZ, SZ)).flatten()
	if(VIS):
		fig, ax = plt.subplots(1, 2)
		ax[0].imshow(np.reshape(features["grayimg"], (SZ, SZ)))
		ax[0].title.set_text("Grayscale")
		ax[1].imshow(np.reshape(features["gradimg"], (SZ, SZ)))
		ax[1].title.set_text("Gradient Image")
		plt.show()


	# 250 x 2 = 500   images
	# 250 x 4 x 4 = 4000 histograms


	return features

img = "Train_Set/Ziwang_Xu_1.jpg"
low_level(cv2.imread(img))
