import numpy as np
import os

lst = np.load('attr_list.npy')
for attr in lst:
	name = '_'.join(attr.split(" "))
	# print(name)
	print(('sbatch run_att_clf.sh "' + name + '" ' + name))
	os.system(('sbatch run_att_clf.sh "' + name + '" ' + name))