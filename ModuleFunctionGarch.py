import scipy as sp
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
import statistics as stat
import statsmodels.distributions as smd
import os
from sklearn.preprocessing import MinMaxScaler
import torch.optim.lr_scheduler as lr_scheduler

def GenerateUnidParameter(paraMax, paraMin, Npara, Disc):
    ParameterDisc=np.zeros(Npara,'f')
    Dp=int(Npara/Disc)
    D=((paraMax-paraMin)/Disc)
    for k in range(Disc):
        ParameterDisc[Dp*k:Dp*(k+1)]= np.random.uniform(paraMin+k*D, paraMin+(k+1)*D, size=(Dp,))
    return ParameterDisc

def FinalData(Nb1, Na1, Na0, alpha0, alpha1, beta1):
    Npara=Nb1*Na1*Na0
    Para=np.zeros((Npara,3),'f')
    for l in range(Na0):
        for i in range(Na1):
            j=i*Nb1+l*Nb1*Na1
            jf=Nb1+i*Nb1+l*Nb1*Na1
            Para[j:jf,0]=alpha0[l]
            Para[j:jf,1]=alpha1[i]
            Para[j:jf,2]=beta1 
    return Para

class Generator(torch.nn.Module):
        def __init__(self, input_neurons, hidden_neurons1, hidden_neurons2, output_neurons ):
            super(Generator, self).__init__()
            self.hidden= nn.Linear(input_neurons, hidden_neurons1)
            self.hiddenM1= nn.Linear(hidden_neurons1, hidden_neurons2)
            self.hiddenM2= nn.Linear(hidden_neurons2, hidden_neurons2)
            self.hiddenM3= nn.Linear(hidden_neurons2, hidden_neurons1)
            #self.Activ =torch.nn.Tanh()
            #self.Activ =torch.nn.ReLU()
            #self.Activ =torch.nn.LeakyReLU(negative_slope=0.001, inplace=False)
            self.Activ =torch.sin
            #self.Activ =torch.sigmoid
            #self.Activ =torch.heaviside(0)
            #self.Activ2=torch.nn.Softmax(1)
            
            self.eps = 1e-20
            
            self.out= nn.Linear(hidden_neurons1, output_neurons)
        def forward(self, x):
            #x = self.bach1(x)
            x = self.hidden(x)
            x = self.Activ(x)
            #x = self.bach2(x)           
            x = self.hiddenM1(x)
            x = self.Activ(x)
            #x = self.bach3(x)           
            x = self.hiddenM2(x)
            x = self.Activ(x)
            #x = self.bach4(x)           
            x = self.hiddenM3(x)            
            x = self.Activ(x)
            #x = self.bach5(x)  
            #x = self.Activ2(x)
            x = self.out(x)
            #x = self.Activ2(x)
            #x = Op(x)
            x = torch.sigmoid(x)
            
            return x
# Define the controller network (to select learning rates)
class Controller(nn.Module):
    def __init__(self, num_choices):
        super(Controller, self).__init__()
        self.num_choices = num_choices
        self.fc = nn.Linear(1, num_choices)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        learning_rate_probs = self.softmax(self.fc(x))
        return learning_rate_probs
    
def CoVxtxtn(xt2,n,n_steps,Npara):
    CoV2xtxtn=torch.zeros(Npara, device='cuda:0')
    for i in range(Npara):
        CoV2xtxtn[i]=((1/(n_steps-1-n))*(torch.sum(xt2[i,:n_steps-n]*xt2[i,n:])-(1/(n_steps-n))*(torch.sum(xt2[i,:n_steps-n])*(torch.sum(xt2[i,n:])))))*(1/(Ex2[i]))
    return CoV2xtxtn

def save_network(network, epoch_label, minibatch):
    save_filename = 'net_{}_{}.pth'.format(epoch_label , minibatch)
    save_path = os.path.join('./GARCH107', save_filename)
    torch.save(network.state_dict(), save_path)
    
def BestFitLine(min, max, x, V2):
    y=torch.zeros(x.size(0),1)
    y[:,0]=V2
    #plt.scatter(x, y , color='red' )
    xplusone = torch.cat( ( torch.ones(x.size(0),1 ),x) , 1 ) 
    R= torch.linalg.lstsq( xplusone ,y )
    R = R[0:xplusone.size(1)]
    Ybestfit=torch.matmul(xplusone,R[0])
    a1=R[0][0]
    a1=round(a1[0].tolist(),4)
    a2=R[0][1]
    a2=round(a2[0].tolist(),4)
    return Ybestfit, a1, a2

def NNData(T, Dt, Para, Npara):
    n_steps=int(T/Dt)
    Ex2=(Para[:,0])/(1-Para[:,1]-Para[:,2])
    Gamma4=3+(6*(Para[:,1])**2)/(1-(Para[:,1])**2-(Para[:,1]+Para[:,2])**2)
    Gamma6=(15*(1-Para[:,1]-Para[:,2])**3*(1+ (3*(Para[:,1]+Para[:,2])/(1-Para[:,1]-Para[:,2]))+(3*(1+2*(Para[:,1]+Para[:,2])/(1-Para[:,1]-Para[:,2]))*(Para[:,2]**2+2*Para[:,1]*Para[:,2]+3*Para[:,1]**2)/(1-3*Para[:,1]**2-2*Para[:,1]*Para[:,2]-Para[:,2]**2))))/(1-15*Para[:,1]**3-9*Para[:,1]**2*Para[:,2]-3*Para[:,1]*Para[:,2]**2-Para[:,2]**3)
    sigma_0=0.1


    Sigma2_tupdate_0=sigma_0*torch.ones(Npara, device='cuda:0')
    Ex22i=torch.zeros(Npara, device='cuda:0')
    Ex4i=torch.zeros(Npara, device='cuda:0')
    Ex6i=torch.zeros(Npara, device='cuda:0')
    for i in range(1, n_steps):
        Zn=torch.normal(0, 1, size=(Npara,1), device='cuda:0')
        xt2update_0=Sigma2_tupdate_0*(Zn[:,0])**2
        Sigma2_tupdate_0=Para[:,0]+Para[:,1]*xt2update_0+Para[:,2]*(Sigma2_tupdate_0)
        Ex22i[:]+=(1/n_steps)*xt2update_0
        Ex4i[:]+=(1/n_steps)*(xt2update_0**2)
        Ex6i[:]+=(1/n_steps)*(xt2update_0**3)
    Zn=torch.normal(0, 1, size=(Npara,1), device='cuda:0')
    xt2update_0=(Sigma2_tupdate_0)*((Zn[:,0])**2)
    Ex22i[:]+=(1/n_steps)*xt2update_0
    Ex4i[:]+=(1/n_steps)*(xt2update_0**2)
    Ex6i[:]+=(1/n_steps)*(xt2update_0**3)    

    ErrerRelG4=(torch.mean(((Gamma4[:]-(Ex4i/Ex22i**2)[:])/Gamma4[:])**2))**(1/2)
    ErrerRelG6=(torch.mean(((Gamma6[:]-(Ex6i[:]/Ex22i[:]**3))/Gamma6[:])**2))**(1/2)

    print('ErrRelGamma4=',ErrerRelG4)
    print('ErrRelGamma6=',ErrerRelG6)
        

    Gamma4=Ex4i/(Ex22i**2)
    Gamma6=Ex6i/(Ex22i**3)

    data=torch.zeros(Npara, 6, device='cuda:0')
    data[:,0]=Para[:,0]
    data[:,1]=Para[:,1]
    data[:,2]=Para[:,2]


    data[:,3]=Ex22i

    data[:,4]=Gamma4

    data[:,5]=Gamma6
    '''
    data[:,3]=Ex2

    data[:,4]=Gamma4

    data[:,5]=Gamma6
    '''

    ScaledData=torch.zeros(Npara, 6, device='cuda:0')
    ScaledData[:,2]=(data[:,0]-min(data[:,0]))/(max(data[:,0])-min(data[:,0]))
    ScaledData[:,0]=(data[:,1]-min(data[:,1]))/(max(data[:,1])-min(data[:,1]))
    ScaledData[:,1]=(data[:,2]-min(data[:,2]))/(max(data[:,2])-min(data[:,2]))

    ScaledData[:,3:]=data[:,3:]
    return ScaledData