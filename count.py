import os

Dir="train_feat"
X=os.listdir(Dir)
print(len(X))
mul,pairs=0,0
dic={}
mx=0
for i in range(2,630):
	dic[i]=0
for i in X:
	try:
		y=os.listdir(Dir+"/"+i)
		j=len(y)
		if len(y)>1:
			mul+=1
			dic[len(y)]+=1
		pairs+=(j*(j-1))/2
	except:
		pass	

print(mul,pairs,dic)		
