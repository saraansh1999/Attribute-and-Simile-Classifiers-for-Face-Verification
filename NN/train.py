import torch
import itertools
import numpy as np
import pickle
from time import time
import sys, os
from matplotlib import pyplot
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler
import torch.optim as optim
from sklearn.metrics import *

def save_model(net, auth_loss, liv_loss, name, args):
	print('==> Saving models ...')
	state = {
			'auth_loss': auth_loss,
			'liv_loss': liv_loss,
			'net_state_dict': net.state_dict()
		}
	state.update(vars(args))
	dir_name = '/'.join(name.split('/')[:-1])
	if not os.path.exists(dir_name):
		print("Creating Directory: ", dir_name)
		os.makedirs(dir_name)
	torch.save(state, str(name) + '.pth')

def train(optimizer, scheduler, dataloader, loss_fn, model, args, phases):
	start_time = time()

	for epoch in range(args.num_epochs):
		print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
		print('-' * 10)


		for phase in phases:

			if phase == 'train':
				model.train()

				epoch_loss = 0.0
				total_samples = 0

				for (batch_idx, (inputs, labels)) in enumerate(dataloader[phase]):

					if torch.cuda.is_available():
						inputs, labels = inputs.cuda(), labels.cuda()
			
					total_samples += labels.shape[0]

					optimizer.zero_grad()

					with torch.set_grad_enabled(True):
						
						output = model(inputs)
						
						# loss
						criterion = loss_fn
						loss = criterion(output, labels)

						epoch_loss += loss
						
						loss.backward()
						optimizer.step()

				scheduler.step()
				print("==================================================================================")
				print("Train Loss: ", epoch_loss / batch_idx)
				if (epoch + 1) == args.num_epochs:
					save_name = os.path.join(os.path.join(args.save_path, 'models'), str(epoch))
					save_model(model, epoch_loss / batch_idx, None, save_name, args)

				sys.stdout.flush()

			elif phase == 'test' and (epoch + 1) == args.num_epochs:
				model.eval()

				total_pred, total_labels = np.array([]), np.array([])

				for (batch_idx, (inputs, labels)) in enumerate(dataloader[phase]):
					if torch.cuda.is_available():
						inputs, labels = inputs.cuda(), labels.cuda()

					with torch.set_grad_enabled(False):
						output = model(inputs)
						pred = torch.argmax(output, axis=1)

						if total_pred.shape[0]:
							total_pred = np.hstack((total_pred, pred.cpu().numpy()))
							total_labels = np.hstack((total_labels, labels.cpu().numpy()))
						else:
							total_pred = pred.cpu().numpy()
							total_labels = labels.cpu().numpy()

				print("Confusion Matrix -> ")
				print(confusion_matrix(total_labels, total_pred))
				print(classification_report(total_labels, total_pred))


	sys.stdout.flush()