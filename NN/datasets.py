import os, pickle
import numpy as np
from PIL import Image, ImageFile
from random import shuffle

from torch.utils.data import Dataset
from torchvision import datasets, transforms

def get_name(file):
	name = file.split('.')[-2]
	id = int(name.split('_')[-1])
	name = '_'.join(name.split('_')[:-1] + [str(id)])
	return name

class MyDataset(Dataset):
	def __init__(self, data_dir, type, args, sample_thresh=None):
		self.data_dir = data_dir
		self.type = type
		(self.emb, self.labels) = ([], [])
		with open(args.att_file, 'rb') as f:
			Attributes = pickle.load(f)

		for root, dirs, files in os.walk(self.data_dir):
			for file in files:
				fpath = os.path.join(root, file)
				name = get_name(file)
				try:
					self.labels.append(float(Attributes[name][args.trait]) > 0)
					self.emb.append(fpath)
				except:
					pass

		self.emb, self.labels = np.array(self.emb), np.array(self.labels)

		if self.type == "train":
			self.check_unbalance(sample_thresh)

		self.dataset_size = len(self.labels)

		print(type, self.data_dir, self.dataset_size)
		print(type, "Positive -> ", self.labels[self.labels == 1].shape[0])
		print(type, "Negative -> ", self.labels[self.labels == 0].shape[0])

	def check_unbalance(self, thresh):
		ones = np.where(self.labels == 1)[0]
		zeros = np.where(self.labels == 0)[0]
		if ones.shape[0] / zeros.shape[0] < thresh:
			zeros = np.random.permutation(zeros)[:np.int(ones.shape[0] / thresh)]
		elif zeros.shape[0] / ones.shape[0] < thresh:
			ones = np.random.permutation(ones)[:np.int(zeros.shape[0] / thresh)]
		self.emb = np.hstack((self.emb[ones], self.emb[zeros]))
		self.labels = np.hstack((self.labels[ones], self.labels[zeros]))

	def __len__(self):
		return self.dataset_size

	def __getitem__(self, idx):
		features = np.load(self.emb[idx], allow_pickle=True)
		inp = []
		for k in features:
			inp.extend(features[k])
		inp = np.array(inp).astype(np.float32)
		return inp, self.labels[idx].astype(int)

class CNNDataset(Dataset):
	def __init__(self, data_dir, type, args, sample_thresh=None):
		self.transforms = transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.ToTensor()
			])

		self.data_dir = data_dir
		self.type = type
		(self.images, self.labels) = ([], [])
		with open(args.att_file, 'rb') as f:
			Attributes = pickle.load(f)

		for root, dirs, files in os.walk(self.data_dir):
			for file in files:
				fpath = os.path.join(root, file)
				name = get_name(file)
				try:
					self.labels.append(float(Attributes[name][args.trait]) > 0)
					self.images.append(fpath)
				except:
					pass

		self.images, self.labels = np.array(self.images), np.array(self.labels)

		if self.type == "train":
			self.check_unbalance(sample_thresh)

		self.dataset_size = len(self.labels)

		print(type, self.data_dir, self.dataset_size)
		print(type, "Positive -> ", self.labels[self.labels == 1].shape[0])
		print(type, "Negative -> ", self.labels[self.labels == 0].shape[0])

	def check_unbalance(self, thresh):
		ones = np.where(self.labels == 1)[0]
		zeros = np.where(self.labels == 0)[0]
		if ones.shape[0] / zeros.shape[0] < thresh:
			zeros = np.random.permutation(zeros)[:np.int(ones.shape[0] / thresh)]
		elif zeros.shape[0] / ones.shape[0] < thresh:
			ones = np.random.permutation(ones)[:np.int(zeros.shape[0] / thresh)]
		self.images = np.hstack((self.images[ones], self.images[zeros]))
		self.labels = np.hstack((self.labels[ones], self.labels[zeros]))

	def __len__(self):
		return self.dataset_size

	def __getitem__(self, idx):
		img = Image.open(self.images[idx])
		img = self.transforms(img)
		return img, self.labels[idx].astype(int)