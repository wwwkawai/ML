import numpy as np
import scipy.special as ss
class logistic_regression:
    def __init__(self,xdim,learningrate):
        self.xdim=xdim
        self.learningrate=learningrate
        self.w=np.random.normal(0.0,pow(self.xdim,-0.5),(self.xdim,1))
    def train(self,x,y):
        for i in range(0,x.shape[0]):
            xi=x[i].reshape(1,self.xdim)
            yi_output=((self.w).T).dot(xi.T)
            out=ss.expit(yi_output)
            delta=(y[i]-out)*xi
            self.w=self.w+self.learningrate*delta.T
    def test(self,x):
        for xi in x:
            xi=xi.reshape(1,self.xdim)
            yi_output=((self.w).T).dot(xi.T)
            out=ss.expit(yi_output)
            yi=1-out<out
            print(yi)
