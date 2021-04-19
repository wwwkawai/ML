import numpy as np
class perceptron:
    def __init__(self,xdim,learningrate):
        self.xdim=xdim
        self.learningrate=learningrate
        self.w=np.random.normal(0.0,pow(self.xdim,-0.5),(self.xdim,1))
    def train(self,x,y):
        for i in range(0,x.shape[0]):
            xi=x[i].reshape(1,self.xdim)
            yi_output=((self.w).T).dot(xi.T)
            if(np.sign(yi_output)!=np.sign(y[i])):
                self.w=self.w+self.learningrate*y[i]*xi.T
    def test(self,x):
        for xi in x:
            print(np.sign(self.w.T.dot(xi.T)))
x=np.random.normal(0.7,0.5,(1000,2))
y=[]
for i in range(0,1000):
    y.append(x[i][1]-x[i][0])
t=np.array([[0,0.1],[1.1,0],[-1.2,-1],[2,2],[0.1,0],[0.2,0.2],[0.3,0.3],[1,1],[1,0.5]])
xdim=2
learningrate=0.5
model=perceptron(xdim,learningrate)
model.train(x,y)
model.test(t)
            
