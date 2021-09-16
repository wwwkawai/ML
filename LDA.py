import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def LDA(x,y):
    n0=0;
    u0=np.zeros(x.shape[1])
    n1=0;
    u1=np.zeros(x.shape[1])
    for i in range(x.shape[0]):
        if(y[i]==0):
            n0+=1
            u0+=x[i]
        else:
            n1+=1
            u1+=x[i]
    u0/=n0
    u1/=n1
    sigma0=np.zeros((x.shape[1],x.shape[1]))
    sigma1=np.zeros((x.shape[1],x.shape[1]))
    Sb=(u0-u1).T.dot(u0-u1)
    for i in range(x.shape[0]):
        diff0=(x[i]-u0).reshape((1,x.shape[1]))
        diff1=(x[i]-u1).reshape((1,x.shape[1]))
        if(y[i]==0):
            sigma0+=diff0.T.dot(diff0)
        else:
            sigma1+=diff1.T.dot(diff1)
    Sw=sigma0+sigma1
    print(sigma0)
    print(sigma1)
    Sw_inv=np.linalg.pinv(Sw)
    '''
    u,e,v=np.linalg.svd(Sw)
    Sw_inv=v.T*np.linalg.inv(np.diag(e))*u.T
    '''
    w=Sw_inv.dot(u0.T-u1.T)
    return w,u0,u1
'''
data=pd.read_table(r'ww.txt',delimiter=',')
data=np.array(data)
x=np.array(data[:,0:2],dtype=float)
y=np.array(data[:,2:],dtype=int).reshape(data.shape[0])
w,u0,u1=LDA(x,y)
print(w)
'''
x=np.random.randint(10,size=(1000,2))
y=np.zeros(1000)
for i in range(0,10):
    if(x[i][1]-x[i][0]>0):
        y[i]=1
    else:
        y[i]=0

w,u0,u1=LDA(x,y)
v0=u0.dot(w)
v1=u1.dot(w)
xt=np.random.randint(10,size=(10,2))
print(xt)
for i in xt:
    px=i.dot(w)
    print(abs(px-v1)<abs(px-v0))
