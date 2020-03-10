import numpy as np 
import os
import pickle

file1 = open('./lfw_attributes.txt', 'r') 
tags=[]
Dict={}
idx=0
for line in file1:
	if idx == 0:
		idx+=1
		continue	
	arr=line.strip('\n').split('	')

	if idx==1:
		tags=arr
	else:
		name=arr[0].replace(' ','_')+'_'+arr[1]
		Dict[name]={}
		# if(arr[0]=='Jane Fonda'):
		# 	print(name)
		for i in range(2,len(arr)):
			Dict[name][tags[i+1]]=arr[i]
		# break		
	idx+=1

Dict['Tags']=tags

# print(Dict['Hamid_Karzai_11']['Bushy Eyebrows'])

file1.close()

with open('filename.pickle', 'wb') as handle:
    pickle.dump(Dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


np.savez("./attributes", **Dict)  


