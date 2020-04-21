import os
import sys
import numpy as np


file1 = open('./lfw_attributes.txt', 'r') 
idx=0
for line in file1:
	if idx == 0:
		idx+=1
		continue
	if idx==1:	
		arr=line.strip('\n').split('	')
		idx+=1
	else:
		break	

arr=arr[3:]
file1.close()

for i in arr:
	os.system("python3 Net.py "+i)
