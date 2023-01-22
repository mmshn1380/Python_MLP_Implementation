import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd

class MLP:
  def __init__(self,learning_rate,activation):
    self.layers=[]
    self.weights=[0]
    self.bias=[]
    self.normalize=False
    self.learning_rate=learning_rate
    if activation=='sigmoid_unipolar':
      self.activation=self.sigmoid_unipolar
      self.activation_dot=self.sigmoid_unipolar_dot
    elif activation=='sigmoid_bipolar':
      self.activation=self.sigmoid_bipolar
      self.activation_dot=self.sigmoid_bipolar_dot
    elif activation=='relu':
      self.activation=self.reLU
      self.activation_dot=self.reLU_dot

  def add_Input_layer(self,n,bias=True):
    self.layers.append(np.zeros((n+(0 if bias==False else 1),1)))
    self.bias.append(bias)
  def add_Hidden_layer(self,n,std=0.03,bias=True):
    self.weights.append(np.random.normal(0,std,size=(n,self.layers[-1].shape[0])))
    self.bias.append(bias)
    self.layers.append(np.zeros((n+(0 if bias==False else 1),1)))
  def add_Output_layer(self,n,std=0.03):
    self.weights.append(np.random.normal(0,std,size=(n,self.layers[-1].shape[0])))
    self.layers.append(np.zeros((n,1)))
    self.bias.append(False)

  def sigmoid_unipolar(self,val):
    return 1/(1 + np.exp(-val)) 

  def sigmoid_unipolar_dot(self,val):
    return val*(1-val)

  def sigmoid_bipolar(self,val):
    return 2/(1 + np.exp(-val)) - 1

  def sigmoid_bipolar_dot(self,val):
    return 1/2*(1-val*val)

  def reLU(self,val):
    return np.maximum(val,0)

  def reLU_dot(self,val):
    val=self.reLU(val)
    val[val!=0]=1
    return val

  def calculate_output(self,X):
    o=X
    if(self.bias[0]==True):
      o=np.hstack((o,np.ones((o.shape[0],1))))
    for i in range(1,len(self.layers)):
      o=self.activation(np.dot(o,np.transpose(self.weights[i])))
      if(self.bias[i]==True):
        o=np.hstack((o,np.ones((o.shape[0],1))))
    return o

  def update_weight(self):
    E=0
    for p in range(0,self.X.shape[0]):
      self.layers[0]=self.X[p]
      if(self.bias[0]==True):
          self.layers[0]=np.append(self.layers[0],1)
      for i in range(1,len(self.layers)):
          self.layers[i]=self.activation(np.dot(self.weights[i],self.layers[i-1]))
          if(self.bias[i]==True):
              self.layers[i]=np.append(self.layers[i],1)
      E+=1/2*np.dot(self.Y[p]-self.layers[-1],self.Y[p]-self.layers[-1])
      w_copy=self.weights.copy()
      delta=[]
      delta.insert(0,(self.Y[p]-self.layers[-1])*self.activation_dot(self.layers[-1]))
      for i in range(len(self.layers)-2,0,-1):
          delta.insert(0,self.activation_dot(self.layers[i])*np.dot(np.transpose(w_copy[i+1]),delta[0]))
          if(self.bias[i]==True):
            delta[0]=delta[0][:-1]
      delta.insert(0,0)
      for i in range(1,len(self.layers)-1):
          sigma_delta=np.dot(np.transpose(w_copy[i+1]),delta[i+1])
          sigma_delta*=self.activation_dot(self.layers[i])
          if(self.bias[i]==True):
            sigma_delta=sigma_delta[:-1]
          sigma_delta=sigma_delta[np.newaxis].T
          self.weights[i]+=self.learning_rate*np.dot(sigma_delta,self.layers[i-1][np.newaxis])
      self.weights[-1]+=self.learning_rate*np.dot((self.activation_dot(self.layers[-1])*(self.Y[p]-self.layers[-1])).reshape(self.layers[-1].shape[0],1) ,
                                                    self.layers[-2][np.newaxis])  
    return E

  def train(self,X,Y,Emax=20,Kmax=300,normalize=False):
    self.X=X
    self.Y=Y
    self.normalize=normalize
    if normalize:
      self.scaler = StandardScaler()
      self.scaler.fit(self.X)
      self.X=self.scaler.transform(self.X)
    reported_E=[]
    for k in range(0,Kmax):
      E=self.update_weight()
      o=self.calculate_output(self.X)
      MSE=np.square(np.subtract(self.Y.reshape(o.shape[0],o.shape[1]),o)).mean()
      reported_E.append(MSE)
      print(k,"th training cycle =======> MSE Error : ",MSE)
      if(E<Emax):
        break
    pd.DataFrame(reported_E).plot()

  def test(self,test_data_X,test_data_Y):
    if self.normalize:
      test_data_X=self.scaler.transform(test_data_X)
    o=self.calculate_output(test_data_X)
    return np.square(np.subtract(test_data_Y.reshape(o.shape[0],o.shape[1]),o)).mean()
    
data=sklearn.datasets.fetch_california_housing()
X=data['data']
Y=data['target']
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.1, shuffle=True)
 
NN=MLP(learning_rate=0.0005,activation='relu')
NN.add_Input_layer(8)
NN.add_Hidden_layer(32)
NN.add_Hidden_layer(32)
NN.add_Hidden_layer(16)
NN.add_Output_layer(1)
NN.train(X_train,y_train,Emax=1,Kmax=50,normalize=True)