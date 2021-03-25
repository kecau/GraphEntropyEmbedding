#Importing packages#
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#Loading data#

def Loadingdata(City):
    X = pd.read_csv(City+'.csv', header = None).values
    return X

#Calculating the vertex entropy#

def entropy(X):
    E = []
    for i in range(X.shape[0]):
        P = []
        for j in range(X.shape[1]):
            if i !=j:
                e = -X[i][j]*np.log(X[i][j])
                P.append(e)
        P = np.array(P)
        E.append(np.sum(P))
    return np.array(E)

#Calculating the graph entropy#

def graphentropy(X):
    E = []
    for i in range(X.shape[0]):
        e = entropy(X[i])
        E.append(np.sum(e))
    return np.array(E)

#Entropy similarity#
def distance(x,y):
    distance = np.mean(np.power((x - y),2))
    return distance

#Constructing the supervised graph#

def getMatrix(data,E):
    Matrix = []
    for i in range(E.shape[0]):
        dis = []
        for j in range(E.shape[0]):
            dis.append(distance(E[i],E[j]))
        dis = np.array(dis)
        index = np.argsort(dis)[1]
        Matrix.append(data[index])
    return np.array(Matrix)

def Getting_data(City):
    X = Loadingdata(City)
    E = graphentropy(X)
    MatrixX = getMatrix(X,E)
    return X, MatrixX

X,MatrixX = Getting_data(City)

#Training dataset and Test dataset #
TrainXTensor = torch.from_numpy(X.reshape(X.shape[0],18*18)[:90]).type(torch.FloatTensor)
TrainSTensor = torch.from_numpy(MatrixX.reshape(X.shape[0],18*18)[:90]).type(torch.FloatTensor)

TestXTensor = torch.from_numpy(X.reshape(X.shape[0],18*18)[90:]).type(torch.FloatTensor)
TestSTensor = torch.from_numpy(MatrixX.reshape(X.shape[0],18*18)[90:]).type(torch.FloatTensor)

#Defining the loss function#

class My_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x,de1,s,de2,ed1,ed2):
        l1 = torch.mean(torch.pow((x - de1),2))
        l2 = torch.mean(torch.pow((s - de2),2))
        l3 = torch.mean(torch.pow((ed1 - ed2),2))
        loss = l1 + l2 + l3
        return loss
    
#Defining the Embedding model#
class Graph2Vec(nn.Module):
    def __init__(self, n_input,n_output):
        super(Graph2Vec,self).__init__()
        
        self.encode1 = nn.Sequential(nn.Linear(n_input,10),
                                    nn.Dropout(0.7),
                                    nn.Tanh(),
                                    
                                    nn.Linear(10,9),
                                    nn.Dropout(0.7),
                                    nn.Tanh(),
                                     
                                    nn.Linear(9,9),
                                     nn.Dropout(0.7),
                                     nn.Tanh(),
                                     
                                     nn.Linear(9,10),
                                     nn.Dropout(0.7),
                                     nn.Tanh(),
                                    )
        self.decode1 = nn.Sequential(nn.Linear(10,9),
                                    nn.Dropout(0.7),
                                    nn.Tanh(),
                                    
                                    nn.Linear(9,9),
                                    nn.Dropout(0.7),
                                    nn.Tanh(),
                                    
                                    nn.Linear(9,10),
                                    nn.Dropout(0.7),
                                    nn.Tanh(),
                                    
                                    nn.Linear(10,n_output),)
        
    def forward(self,x,matrix):
        ed1 = self.encode1(x)
        ed2 = self.encode1(matrix)
        
        de1 = self.decode1(ed1)
        de2 = self.decode1(ed2)
        
        return ed1,ed2,de1,de2

#Training model#    
Graph2Vecmodel = Graph2Vec(TrainXTensor.shape[1],TrainXTensor.shape[1])
optimizer = torch.optim.Adam(Graph2Vecmodel.parameters(),lr = 0.001)
loss_func = My_Loss()

Loss = []
Loss1 = []
for i in range(500):
    ed1,ed2,de1,de2 = Graph2Vecmodel(TrainXTensor,TrainSTensor)
    loss = loss_func(TrainXTensor,de1,TrainSTensor,de2,ed1,ed2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch: ',i, '|' ,'Traininng loss is %.4f' %loss.data.numpy())
#Testing the model#
ted1,ed2,de1,de2 = Graph2Vecmodel(TestXTensor,TestSTensor)

#Saving the results#
def Savingdata(City, Embed):
    np.savetxt('EV'+City+'.csv', Embed, delimiter = ',')
    return None