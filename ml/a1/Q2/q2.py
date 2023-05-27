import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys
plt.rcParams['figure.figsize']=(10,8)
pre_dir=sys.argv[1]
curr_dir=os.getcwd()
os.chdir(pre_dir)
m=1000000
x0=np.ones((m,1))
x1=np.random.normal(3,4,m)
x1=x1.reshape(-1,1)
x2=np.random.normal(-1,4,m)
x2=x2.reshape(-1,1)

X=np.append(x0,x1,axis=1)
X=np.append(X,x2,axis=1)

theta=np.array([3,1,2])
theta=theta.reshape(-1,1)

E=np.random.normal(0,math.sqrt(2),m).reshape(-1,1)


Y=np.dot(X,theta)+E

tm=np.append(X,Y,axis=1)
np.random.shuffle(tm)
X=tm[:,:3]
Y=tm[:,-1]
Y=Y.reshape(-1,1)

batch_len=[1,100,10000,1000000]

bl=batch_len[0]
batch=[(X[i:i+bl],Y[i:i+bl]) for i in range(0,m,bl)]

# ### SGD

theta=np.zeros((3,1))

def hypotesis(theta,X):
    return np.dot(X,theta)

def cost_derivative(theta,X,Y):
    temp=np.append(np.sum((Y-hypotesis(theta,X))*X[:,0:1]),np.sum((Y-hypotesis(theta,X))*X[:,1:2]))
    temp=np.append(temp,np.sum((Y-hypotesis(theta,X))*X[:,2:3]))
    temp=temp.reshape(-1,1)
    return temp

def cost(theta,X,Y):
    return (0.5/len(X))*np.sum((Y-hypotesis(theta,X))**2)

def summation(Theta,x0,x1,x2,y,j):
    sm=0
    m=len(x0)
    for i in range(m):
        sm+=(Theta[0][0]*x0[i][0]+Theta[1][0]*x1[i][0]+Theta[2][0]*x2[i][0]-y[i][0])*(x0[i][0] if j==0 else x1[i][0] if j==1 else x2[i][0])
    return sm/m

l_rate=0.001
itr=0
ini_cost,final_cost=1,2
th_list=[[0],[0],[0]]
cnt=0
while abs(ini_cost-final_cost)>1e-11:
    avg=0
    ini_cost=cost(theta,X,Y)
    for i in batch: #some breking condition like cost>1e-5
        x_i=i[0]
        y_i=i[1]
        Theta=theta
        x_0,x_1,x_2=x_i[:,0:1],x_i[:,1:2],x_i[:,2:3]
        t0=Theta[0]-l_rate*(summation(Theta,x_0,x_1,x_2,y_i,0))
        t1=Theta[1]-l_rate*(summation(Theta,x_0,x_1,x_2,y_i,1))
        t2=Theta[2]-l_rate*(summation(Theta,x_0,x_1,x_2,y_i,2))
        
        theta=np.array([t0,t1,t2]).reshape(-1,1)
        avg+=cost(theta,x_i,y_i)
        if cnt%10==0:
            th_list[0].append(float(t0))
            th_list[1].append(float(t1))
            th_list[2].append(float(t2))
        cnt+=1
    itr+=1
    final_cost=cost(theta,X,Y)
 #   print(avg/len(batch),' ',abs(ini_cost-final_cost))
    
#print('final theta is :  ',theta)
# cost(theta,X,Y)

temp = np.genfromtxt('X.csv',dtype=float ,delimiter=',')
# print(temp)
os.chdir(curr_dir)
testcase_x1 = temp[0:, 0]
testcase_x1=testcase_x1.reshape(-1,1)
testcase_x2 = temp[0:, 1]
testcase_x2 = testcase_x2.reshape(-1,1)
# testcase_Y= temp[1:, 2]
# testcase_Y=testcase_Y.reshape(-1,1)
X0 = np.ones((testcase_x1.shape))
XT = np.append(X0,testcase_x1, axis=1)
XT = np.append(XT, testcase_x2, axis=1)
def predict(XT,theta):
    final_y=np.dot(XT,theta)
    return final_y
final_y=predict(XT,theta)
f=open('result_2.txt','w')
for i in final_y:
    f.write(str(i[0])+'\n')
f.close()
# print('diffrence in cost between traind sample data and given data  ' ,end=' ')
# error = cost(theta,X,Y)-cost(theta,XT,testcase_Y)
# print(error)

# print('cost by given hypothesis ' ,end=' ')
# print(cost(np.array([3,1,2]).reshape((-1,1)),XT,testcase_Y))

ax=plt.axes(projection='3d')
ax.set_title('movement of theta with batchsize 1')
ax.plot(np.array(th_list[0]),np.array(th_list[1]),np.array(th_list[2]),'o-',color='red')
ax.set_xlabel('theta 0')
ax.set_ylabel('theta 1')
ax.set_zlabel('theta 2')
#plt.show()
plt.savefig('plot_batch_size_1.png')
