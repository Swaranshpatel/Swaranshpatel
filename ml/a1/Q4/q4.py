import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
traning_data=sys.argv[1]
prediction_data=sys.argv[2]
curr_dir=os.getcwd()
os.chdir(traning_data)
plt.rcParams['figure.figsize']=(10,8)

X = np.genfromtxt('X.csv', delimiter=',')
# X = np.loadtxt('data/q4/q4x.dat')
Y = np.loadtxt('Y.csv', dtype=str).reshape(-1,1)

# normalization
mean = np.mean(X, axis=0)
mean=mean.reshape(1,-1)
var = np.std(X, axis=0)
var=var.reshape(1,-1)
X-= mean
X/= var

Y[Y=='Alaska']=1
Y[Y=='Canada']=0
phi=np.sum([Y=='1'])/X.shape[0]
# print('phi : ',phi)

idx_0,idx_1=[i for i in range(len(Y)) if Y[i][0]=='0'],[i for i in range(len(Y)) if Y[i][0]=='1']

mean0,mean1=(np.mean(X[idx_0], axis=0)).reshape(1,-1),(np.mean(X[idx_1], axis=0)).reshape(1,-1)
#print(mean0,mean1)

X[1].reshape(-1,2)

def cov(X,Y,m0,m1):
    c0=np.zeros((2,2))
    c1=np.zeros((2,2))
    even_count=0; odd_count=0
#     print(m0)
    for i in range(len(X)):
        if not (Y[i]=='1'):
            c0=c0+np.dot((X[i].reshape(-1,2)-m0).T, (X[i].reshape(-1,2)-m0))
            odd_count+=1
#             print(c0)
        else:
            c1=c1+np.dot((X[i].reshape(-1,2)-m1).T, (X[i].reshape(-1,2)-m1))
            even_count+=1
#             print(c1)
    c=(c0+c1)/len(X)
    c0/=odd_count
    c1/=even_count
    return c,c0,c1

c,c0,c1=cov(X,Y,mean0,mean1)
#print(c0,c1)

plt.scatter(X[idx_0,0],X[idx_0,1],marker='o',c='r')
plt.scatter(X[idx_1,0],X[idx_1,1],marker='s',c='b',s=40)
# plt.show()
temp,cinv =math.log(phi/(1-phi)), np.linalg.inv(c)
m,c=np.dot(cinv, (mean1-mean0).T), (1/2)*((np.dot(np.dot(mean1, cinv), mean1.T))-(np.dot(np.dot(mean0, cinv), mean0.T)))
xv = np.array([np.min(X[:, 1] -1 ), np.max(X[:, 1] +1 )])
xv=xv.reshape(1,-1)
y = np.dot((-1./m[0:1]),np.dot(m[1:2], xv)) - c
os.chdir(curr_dir)

plt.title('Training Data with Linear and Quadratic Decision Boundary')
plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(X[idx_0,0],X[idx_0,1],marker='o',c='r')
plt.scatter(X[idx_1,0],X[idx_1,1],marker='s',c='b',s=40)
plt.plot(xv.ravel(),y.ravel())
plt.savefig('Training Data with Linear  Decision Boundary')
# plt.show()

#print(xv,y)
slope=(y[0][0]-y[0][1])/(xv[0][0]-xv[0][1])
#print(slope)


c0_inv,c0_det = np.linalg.inv(c0),np.linalg.det(c0)
c1_inv,c1_det = np.linalg.inv(c1),np.linalg.det(c1)

a,b = c1_inv - c0_inv,2* (np.dot(c1_inv, mean1.T) - np.dot(c0_inv,mean0.T))
t1=np.dot(np.dot(mean1, c1_inv),mean1.T)
t2=np.dot(np.dot(mean0, c0_inv),mean0.T)
t3=math.log(c0_det/c1_det)
d = t1 - t2- t3

x2_val,x1_val=[], np.linspace((np.min(X[:,1])-1), (np.max(X[:,1])+1), 100)
c1 = a[1,1]
for i in range(len(x1_val)):
    t1=(b[0,0]*x1_val[i]) + d[0,0]
    t2 ,t4= (a[0,0]*(x1_val[i]**2)) - t1, b[1,0]
    t3 = ((a[0,1]+a[1,0])*x1_val[i]) -t4
    x2_val.append(np.roots([c1, t3, t2]))

y_q0= [(x2_val[i][0]) for i in range(len(x2_val))]
y_q1= [(x2_val[i][1]) for i in range(len(x2_val))]
os.chdir(curr_dir)
plt.title('Training Data with Linear and Quadratic Decision Boundary')
plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(X[idx_0,0],X[idx_0,1],marker='o',c='r')
plt.scatter(X[idx_1,0],X[idx_1,1],marker='s',c='b',s=40)
plt.scatter(x1_val,y_q1,marker='^')
plt.scatter(x1_val,y_q0)
plt.plot(xv.ravel(),y.ravel(),c='black')
plt.ylim(bottom=-4)
plt.ylim(top=6)
plt.savefig('Training Data with Linear and Quadratic Decision Boundary') 
# plt.show()


os.chdir(prediction_data)
preX= np.genfromtxt('X.csv', delimiter=',')
os.chdir(curr_dir)
# print(preX)
f=open('result_4.txt','w')
for i in preX:
    f.write('Alaska'+'\n' if slope>=(i[0]/i[1]) else 'canada'+'\n')
f.close()
