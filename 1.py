#simple neural network's implementation
import numpy as np
import pandas as pd
import scipy.special
def loaddata(filename):
    data=pd.read_csv(filename)
    datamat=np.array(data)
    dataMat=datamat[1:,0:]
    return dataMat
class nn:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.wi_h=np.random.normal(0.0,pow(self.inodes,-0.5),(self.hnodes,self.inodes))
        self.wh_o=np.random.normal(0.0,pow(self.hnodes,-0.5),(self.onodes,self.hnodes))
        self.lr=learningrate
        self.activation_func=lambda x:scipy.special.expit(x)
    def train(self,inputs_list,labels_list):
        inputs=np.array(inputs_list,ndmin=2).T
        targets=np.array(labels_list,ndmin=2).T
        hidden_inputs=np.dot(self.wi_h,inputs)         #forward probagation
        hidden_outputs=self.activation_func(hidden_inputs)
        final_inputs=np.dot(self.wh_o,hidden_outputs)
        final_outputs=self.activation_func(final_inputs    #calculate partial derivative
        out_put_error=targets-final_outputs
        hidden_error=np.dot(self.wh_o.T,out_put_error)
        self.wh_o+=self.lr*np.dot((out_put_error*final_outputs*(1-final_outputs)),np.transpose(hidden_outputs))  #renew weight
        self.wi_h+=self.lr*np.dot((hidden_error*hidden_outputs*(1-hidden_outputs)),np.transpose(inputs))
        #print(np.argmax(final_outputs)==np.argmax(labels_list))
    def test(self,inputs_list):
        inputs=inputs=np.array(inputs_list,ndmin=2).T
        hidden_inputs=np.dot(self.wi_h,inputs)
        hidden_outputs=self.activation(hidden_inputs)
        final_inputs=self.np.dot(self.wh_o,hidden_outputs)
        final_outputs=self.activation_func(final_inputs)
        return np.argmax(final_outputs)
input_nodes=784
hidden_nodes=500
output_nodes=10
learning_rate=0.1
model=nn(input_nodes,hidden_nodes,output_nodes,learning_rate)
datamat=loaddata(#filename)
epochs=10
for e in range(epochs):
    for data in datamat:
        label_list=np.zeros(output_nodes)
        label_list[data[0]]=1
        input_list=data[1:]/255
        model.train(input_list,label_list)
