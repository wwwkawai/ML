import numpy as np
def LSE(x,y):
    return (np.linalg.pinv(x.T.dot(x)).dot(x.T.dot(y)))
def LSE_regular(x,y,lamda):
    return (np.linalg.inv(x.T.dot(x)+lamda*np.identity(x.shape[1])).dot(x.T.dot(y)))
x=np.array([[1,1],[1,1]])
y=np.array([[3,4]])
y=y.T
print(LSE(x,y))
print(LSE_regular(x,y,0.05))
