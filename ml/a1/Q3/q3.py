import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(10,8)
# %matplotlib notebook
traning_data=sys.argv[1]
prediction_data=sys.argv[2]
curr_dir=os.getcwd()
os.chdir(traning_data)

tempX = np.genfromtxt('X.csv', delimiter=',')
Y=np.genfromtxt('Y.csv', delimiter=',')

Y=Y[:].reshape((-1,1))

x1=tempX[:,0].reshape((-1,1))
x2=tempX[:,1].reshape((-1,1))

X=np.append(x1,x2,axis=1)
x0=np.ones((x1.shape[0],1))

os.chdir(curr_dir)

# ax=plt.axes(projection='3d')
# ax.scatter(x1,x2,Y,'o-',color='red')
# plt.show()
# ## normalization

m=np.mean(X)
sd=np.std(X)
X-=m
X/=sd

X=np.append(x0,X,axis=1)

theta=np.zeros((X.shape[1],1)) 
hypo=np.dot(X,theta)
new_hypo=(1/(1 + np.exp(-hypo)))

g=np.dot(X.T,(new_hypo-Y))
I = np.identity(X.shape[0])
diag=I * np.dot(new_hypo.T,(1-new_hypo))
h=np.dot(X.T, np.dot(diag, X))    #H=trans(X).S.X and S=diag(m(1-m)) where m=j_theta

final_theta=theta-np.dot(np.linalg.inv(h),g)

#print('final theta value is  :  ', final_theta)

def find_line(theta,X):
    y=np.dot(X,theta)
#     print(y)
    return y

lin=find_line(final_theta,X)

def find_idx(X,Y):
    idx0,idx1=[],[]
    for i in range(len(X)):
#         print(Y[i])
        if Y[i]:
            idx1.append(i)
        else:
#             print(Y[i])
            idx0.append(i)
#             print(i)
#     print(idx0,idx1)
    return idx0,idx1

idx0,idx1=find_idx(X,Y)
plt.title('decision boundary fit by logistic regression')
plt.scatter(X[idx0,1],X[idx0,2],c='r')
plt.scatter(X[idx1,1],X[idx1,2],c='b')
x_val = np.array([np.min(X[:, 1] ), np.max(X[:, 1] )])
x_val=x_val.reshape(1,-1)
y_val = np.dot((-1./final_theta[2:3]),np.dot(final_theta[1:2], x_val)) - final_theta[0:1]
plt.xlabel('X1')
plt.ylabel('X2')
plt.plot(x_val[0],y_val[0])
#print(x_val,y_val)
plt.savefig('plot_scatter_with_boundary ')


os.chdir(prediction_data)
prX = np.genfromtxt('X.csv', delimiter=',')
x1=prX[:,0].reshape((-1,1))
x2=prX[:,1].reshape((-1,1))

# X=np.append(x1,x2,axis=1)
# x0=np.ones((x1.shape[0],1))
# X=np.append(x0,X,axis=1)
D=y_val[0][0]-y_val[0][1]
N=(x_val[0][0]-x_val[0][1])
#print(D,N)
slope=D/N
os.chdir(curr_dir)
#print(slope)
f=open('result_3.txt','w')
for i in range(len(x1)):
    f.write(str( 1 if slope<=(x1[i][0]/x2[i][0]) else 0)+'\n')
f.close()

# y_pre=np.dot(X,final_theta)
# print(y_pre)
# print(X)
# print(theta)
