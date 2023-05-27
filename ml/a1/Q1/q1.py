import sys
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
# %matplotlib notebook
traning_data=sys.argv[1]
prediction_data=sys.argv[2]
curr_dir=os.getcwd()
plt.rcParams['figure.figsize']=(10,8)
os.chdir(traning_data)

a1=pd.read_csv('X.csv',names=['den'])
a2=pd.read_csv('Y.csv',names=['vol'])
os.chdir(curr_dir)
os.chdir(prediction_data)
a3=pd.read_csv('X.csv',header=None)

x=np.array(a1)
y=np.array(a2)
test=np.array(a3)

mean=np.mean(x)
var=np.std(x)
x-=mean
x/=var

os.chdir(curr_dir)
plt.title('scatter plot between X and Y before prediction line')
plt.plot(x,y,'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('plot_scatter')


Theta,l_rate=[0,0],0.01

def cost(Theta,x,y):
    m=len(x)
    j_theta=0
    for i in range(m):
        j_theta+=(Theta[0]+Theta[1]*x[i][0]-y[i][0])**2
    j_theta*=(0.5/m)
#     print(j_theta)
    return j_theta

def hypotheses(theta,xi):
    # print(theta,xi)
    # print(int(theta[0]+theta[1]*xi))
    return int(theta[0]+theta[1]*xi)

def summation(Theta,x,y,j):
    sm=0
    m=len(x)
    for i in range(m):
        sm+=(Theta[0]+Theta[1]*x[i][0]-y[i][0])*(x[i][0] if j else 1)
    return sm/m

# s=1
error=[cost(Theta,x,y)]
te0,te1=[Theta[0]],[Theta[1]]
while error[-1]>1.3*10**-6:
    t1=Theta[1]-l_rate*(summation(Theta,x,y,1))
    t0=Theta[0]-l_rate*(summation(Theta,x,y,0))
    Theta=[t0,t1]
    error.append(cost(Theta,x,y))
    te0.append(Theta[0])
    te1.append(Theta[1])
#     if s%1000==0:
#         print(cost(theta,x,y))
#     s+=1
    
#print(Theta)


def pred(x,Theta):
    li=[]
    for i in x:
        li.append([Theta[0]+Theta[1]*i[0]])
    return li

y_new=pred(x,Theta)

test_result=pred(test,Theta)

plt.title('scatter plot between X and Y with prediction line')
plt.plot(x, y,'o')
plt.plot(x,y_new,color='r')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('plot_scatter_final')

f=open('result_1.txt','w')

for i in test_result:
    f.write(str(i[0])+'\n')

f.close()

# plt.rcParams['figure.figsize']=(8,6)

# ## 3D PLOTING

def fn(theta,x,y):
    F=1
    sm=None
    for i in range(len(x)):
        sm=(theta[0]+theta[1]*x[i][0]-y[i][0])**2 if F else sm+(theta[0]+theta[1]*x[i][0]-y[i][0])**2 
        F=0
    sm*=0.5
    return sm/len(x)

def fn2(theta,x,y):
#     cost(theta,x,y)
    n,m=len(theta[0]),len(theta[0][0])
    zmt=[]
    for i in range(m):
        li=[]
        for j in range(n):
            li.append(cost((theta0[i][j],theta1[i][j]),x,y))
        zmt.append(li)
    return zmt

theta0=np.linspace(0,2,200)
theta1=np.linspace(-1,1,200)
theta0,theta1=np.meshgrid(theta0,theta1)
Z=fn((theta0,theta1),x,y)
Z2=fn2((theta0,theta1),x,y)
for i  in range(len(Z2)):
    for j in range(len(Z2[0])):
        Z[i][j]=Z2[i][j]

# plt.rcParams['figure.figsize']=(15,12)
plt.clf()
ax=plt.axes(projection='3d')
# fig=plt.figure()
ax.plot_surface(theta0,theta1,Z,cmap='rainbow',alpha=0.6)
ax.scatter(te0,te1,error,'o-',color='black',alpha=1)
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('z')
ax.view_init(12,-36)
# plt.show()
plt.savefig('3D error plot.png')


fig,ax=plt.subplots(1,1)
cp = ax.contourf(theta0,theta1,Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('step size 0.01')
ax.plot(te0,te1, marker='s',color ='r')
ax.set_xlabel('theta 0')
ax.set_ylabel('theta 1')
#plt.show()
fig.savefig('contour_0.01.png')



def function_for_contur(l_rate,theta,x,y,tem):
    error=[cost(theta,x,y)]
    te0,te1=[theta[0]],[theta[1]]
    cnt=0
    while error[-1]>1.3*10**-6:
        t1=theta[1]-l_rate*(summation(theta,x,y,1))
        t0=theta[0]-l_rate*(summation(theta,x,y,0))
        theta=[t0,t1]
        error.append(cost(theta,x,y))
        te0.append(theta[0])
        te1.append(theta[1])
        if cnt>tem:
            break
        cnt+=1
    return te0,te1

te0,te1=function_for_contur(0.025,[0,0],x,y,1000)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(theta0,theta1,Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('step size 0.025')
ax.plot(te0,te1, marker='s',color ='r')
ax.set_xlabel('theta 0')
ax.set_ylabel('theta 1')
#plt.show()
fig.savefig('contour_0.025.png')

te0,te1=function_for_contur(0.001,[0,0],x,y,1000)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(theta0,theta1,Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('step size 0.001')
ax.plot(te0,te1, marker='o',color ='r')
ax.set_xlabel('theta 0')
ax.set_ylabel('theta 1')
#plt.show()
fig.savefig('contour_0.001.png')


te0,te1=function_for_contur(0.1,[0,0],x,y,100)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(theta0,theta1,Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('step size 0.1 overflow encounter ')
ax.plot(te0,te1, marker='s',color ='r')
ax.set_xlabel('theta 0')
ax.set_ylabel('theta 1')
#plt.show()
fig.savefig('contour_0.1.png')

