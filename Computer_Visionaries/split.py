import numpy as np 
import os
from shutil import copyfile

try:
	os.mkdir("Test_Set")
except:
	pass

try:
	os.mkdir("Train_Set")
except:
	pass


file1 = open('./peopleDevTest.txt', 'r') 

idx=0
for line in file1:
	if idx == 0:
		idx+=1
		continue	
	arr=line.strip('\n').split('	')
	
	zeros=""
	for rep in range(4-len(arr[1])):
		zeros+="0"

	src="./lfw/"+arr[0]+"/"+arr[0]+"_"+zeros+arr[1]+".jpg"
	dst="./Test_Set/"+arr[0]+"_"+arr[1]+".jpg"

	copyfile(src,dst)

	idx+=1


file1.close()


file1 = open('./peopleDevTrain.txt', 'r') 

idx=0
for line in file1:
	if idx == 0:
		idx+=1
		continue	
	arr=line.strip('\n').split('	')
	
	zeros=""
	for rep in range(4-len(arr[1])):
		zeros+="0"

	src="./lfw/"+arr[0]+"/"+arr[0]+"_"+zeros+arr[1]+".jpg"
	dst="./Train_Set/"+arr[0]+"_"+arr[1]+".jpg"

	copyfile(src,dst)

	idx+=1


file1.close() 