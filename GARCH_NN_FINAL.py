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


import ModuleFunctionGarch
from ModuleFunctionGarch import *  #Fonction module
#from ModuleGeneratorDiscriminator import * #GeneratorDiscrimantor module

'''*********************Data set*********************'''


Npara=40000
alpha0=np.zeros(Npara,'f')
a00 = 10**(-3)
a01  = 10**(-2)

Disc=1000
alpha0=GenerateUnidParameter(a01, a00, Npara, Disc)

alpha1=np.zeros(Npara,'f')
a10 = 0
a11  = 0.3
Disc=1000
alpha1=GenerateUnidParameter(a11, a10, Npara, Disc)
rng = np.random.default_rng(seed=None)
rng.shuffle(alpha1)

beta1=np.zeros(Npara,'f')
b10 = 0
b11  = 0.99
Disc=1000
beta1=GenerateUnidParameter(b11, b10, Npara, Disc)


A=np.delete(alpha1, beta1**3+3*alpha1*beta1**2+9*alpha1**2*beta1+15*alpha1**3>0.97, 0)
B=np.delete(beta1, beta1**3+3*alpha1*beta1**2+9*alpha1**2*beta1+15*alpha1**3>0.97, 0)
A0=np.delete(alpha0, beta1**3+3*alpha1*beta1**2+9*alpha1**2*beta1+15*alpha1**3>0.97, 0)

print(np.size(A))
A1=np.delete(A, 0.97-3*A**2-2*A*B-B**2<0, 0)
B1=np.delete(B, 0.97-3*A**2-2*A*B-B**2<0, 0)
A10=np.delete(A0, 0.97-3*A**2-2*A*B-B**2<0, 0)
print(np.size(A))

Npara=np.size(A1)
Para=torch.zeros(Npara,3, device='cuda:0')
Para[:,0]=torch.as_tensor(A10)
Para[:,1]=torch.as_tensor(A1)
Para[:,2]=torch.as_tensor(B1)



T=1
Dt=10**(-7)
ScaledData=NNData(T, Dt, Para, Npara)

'''Data Validation '''

                        
Na0=10
Na1=20
Nb1=20
Nparatest=Nb1*Na1*Na0

alpha0T=np.zeros(Na0,'f')
Disc=1000
alpha0T=GenerateUnidParameter(a01, a00, Nparatest, Disc)

alpha1T=np.zeros(Na1,'f')
Disc=1000
alpha1T=GenerateUnidParameter(a11, a10, Nparatest, Disc)
rng = np.random.default_rng(seed=None)
rng.shuffle(alpha1T)


beta1T=np.zeros(Nb1,'f')
Disc=1000
beta1T=GenerateUnidParameter(b11, b10, Nparatest, Disc)


AT=np.delete(alpha1T, beta1T**3+3*alpha1T*beta1T**2+9*alpha1T**2*beta1T+15*alpha1T**3>0.97, 0)
BT=np.delete(beta1T, beta1T**3+3*alpha1T*beta1T**2+9*alpha1T**2*beta1T+15*alpha1T**3>0.97, 0)
A0T=np.delete(alpha0T, beta1T**3+3*alpha1T*beta1T**2+9*alpha1T**2*beta1T+15*alpha1T**3>0.97, 0)

print(np.size(AT))
A1T=np.delete(AT, 0.97-3*AT**2-2*AT*BT-BT**2<0, 0)
B1T=np.delete(BT, 0.97-3*AT**2-2*AT*BT-BT**2<0, 0)
A10T=np.delete(A0T, 0.97-3*AT**2-2*AT*BT-BT**2<0, 0)
print(np.size(AT))

Nparatest=np.size(A1T)
ParaTest=torch.zeros(Nparatest,3, device='cuda:0')
ParaTest[:,0]=torch.as_tensor(A10T)
ParaTest[:,1]=torch.as_tensor(A1T)
ParaTest[:,2]=torch.as_tensor(B1T)


ScaledDataTest=NNData(T, Dt, ParaTest, Nparatest)


'''Networks'''
#Generator Network
NetworkG = Generator(input_neurons = 3, hidden_neurons1 = 2048 , hidden_neurons2 = 4096, output_neurons = 3).to('cuda:0')

'''Optimizers'''
#Generator optimizer Network
optimizerG = torch.optim.Adam(NetworkG.parameters(), lr=0.00001, betas=(0.5, 0.999))


Ntrain=Npara  #Size of the training set
batche_size=500

NetworkG.train()
LossMSE=nn.MSELoss()

Err_Validation=torch.zeros(0, device='cuda:0')   #Estimated Probability of having a fake sample 
Err_Training=torch.zeros(0, device='cuda:0')  #Discrimnator array loss 


'''Parttion'''
r=0.01
for epoch in range(20):
    for mini_batches in range(int(Ntrain/batche_size)):
            Xtrain=torch.as_tensor(ScaledData[mini_batches*batche_size:(mini_batches+1)*batche_size,:])
            Xpi=torch.empty(batche_size,3, device='cuda:0')
            Xpi[:,:]=Xtrain[:,3:]
            ParaG=NetworkG(Xpi)
            #breakpoint()
            #Generator_loss= LossMSE(Xtrain[:,0].view(batche_size,1), ParaG)
            Generator_loss= torch.mean((Xtrain[:,2].view(batche_size)-ParaG[:,2])**2)+torch.mean((Xtrain[:,0].view(batche_size)-ParaG[:,0])**2)+torch.mean((Xtrain[:,1].view(batche_size)-ParaG[:,1])**2)+r*(1/2)*(torch.mean(abs(torch.maximum( torch.tensor(0),ParaG[:,1]+ParaG[:,0]-1))**2))                                  
            optimizerG.zero_grad()
            Generator_loss.backward(retain_graph=True)
            #Genrator network parameters update
            optimizerG.step()

    DescriTorch=torch.zeros(1, device='cuda:0')
    GeneraTorch=torch.zeros(1)
    print('epoch=', epoch)
    DescriTorch[0]=Generator_loss.detach()
    Err_Training=torch.cat((Err_Training, DescriTorch), dim=0)

print('grad=', Generator_loss)    
print('grad=',torch.autograd.grad) 
Err_TrainingCpu=Err_Training.cpu()

plt.figure(1)
plt.plot( np.log(Err_TrainingCpu)[:]/np.log(10),'blue', label='training') 
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.legend()
#plt.savefig('Garch_MonteCarlo_Gamma6_Linear_107_n3_Var3__001_105.pdf')
plt.show()
plt.close()

"""""""""""""""""""""""""""""""""""""""""Validation Part"""""""""""""""""""""""""""""""""""""""""

#PATH="./GARCH107/net_0.1_105106_3.pth"
#checkpoint = torch.load(PATH)
#NetworkG.load_state_dict(checkpoint)
#NetworkG.eval()

DataTorch=torch.as_tensor(ScaledDataTest[:,:])
ParaPred=NetworkG(DataTorch[:,3:])
AlphaPredTest=ParaPred[:,0]
BetaPredTest=ParaPred[:,1]
alpha0PredTest=ParaPred[:,2]

Alpha0tPy=torch.zeros(Nparatest,1, device='cuda:0')
Alpha0tPy[:,0]=ParaTest[:,0]
Alpha1tPy=torch.zeros(Nparatest,1, device='cuda:0')
Alpha1tPy[:,0]=ParaTest[:,1]
betaTPy=torch.zeros(Nparatest,1, device='cuda:0')
betaTPy[:,0]=ParaTest[:,2]
AlphaPredTest=AlphaPredTest*(max(Para[:,1])-min(Para[:,1]))+min(Para[:,1])
BetaPredTest=BetaPredTest*(max(Para[:,2])-min(Para[:,2]))+min(Para[:,2])
alpha0PredTest=alpha0PredTest*(max(Para[:,0])-min(Para[:,0]))+min(Para[:,0])

lossRelalpha1=torch.mean((Alpha1tPy-AlphaPredTest)**2/Alpha1tPy**2)

lossRelalpha0=torch.mean((Alpha0tPy-alpha0PredTest)**2/alpha0PredTest**2)

lossRelbeta1=torch.mean((betaTPy-BetaPredTest)**2/BetaPredTest**2)

print('lossRelalpha1=',lossRelalpha1)

print('lossRelbeta1=',lossRelbeta1)

print('lossRelalpha0=',lossRelalpha0)




Y1=np.random.uniform(0, 0.3, 1000)
X1=Y1

alpha1cpu=(Alpha1tPy.detach()).cpu()
alpha1Predcpu=(AlphaPredTest.detach()).cpu()
beta1cpu=(betaTPy.detach()).cpu()
beta1Predcpu=(BetaPredTest.detach()).cpu()
alpha0cpu=(Alpha0tPy.detach()).cpu()
alpha0Predcpu=(alpha0PredTest.detach()).cpu()


Ybestfit, a1, a2=BestFitLine(0, 0.3, alpha1cpu, alpha1Predcpu)

plt.figure(2)
plt.plot((alpha1cpu.numpy())[:], ((alpha1Predcpu).numpy())[:],'o')
plt.plot(X1, Y1, label='Y=X')
plt.plot( alpha1cpu.tolist(), Ybestfit.tolist(), label='Y='+str(a2)+'*X+'+str(a1))
plt.xlabel('$\\alpha_1$ reference')
plt.ylabel('$\\alpha_1$ predicted')
plt.legend()

#plt.savefig('alpha013_Predicted_GARCH_Gamma6_Linear_MonteCarlo_107_Pena1_01_105106_3.pdf')
plt.show()
plt.close()

Y1=np.random.uniform(0, 1, 1000)
X1=Y1
Ybestfit, a1, a2=BestFitLine(0, 1, beta1cpu, beta1Predcpu)

plt.figure(3)
plt.plot((beta1cpu.numpy())[:], ((beta1Predcpu).numpy())[:],'o')
plt.plot(X1, Y1, label='Y=X')
plt.plot( beta1cpu.tolist(), Ybestfit.tolist(), label='Y='+str(a2)+'*X+'+str(a1))

plt.xlabel('$\\beta_1$ reference')
plt.ylabel('$\\beta_1$ predicted')
plt.legend()

#plt.savefig('alpha013_Predicted_GARCH_Gamma6_Linear_MonteCarlo_107_Pena1_01_105106_3.pdf')
plt.show()
plt.close()


Y1=np.random.uniform(0, 0.01, 1000)
X1=Y1
Ybestfit, a1, a2=BestFitLine(0, 0.01, alpha0cpu, alpha0Predcpu)

plt.figure(4)
plt.plot((alpha0cpu.numpy())[:], ((alpha0Predcpu).numpy())[:],'o')
plt.plot(X1, Y1, label='Y=X')
plt.plot( alpha0cpu.tolist(), Ybestfit.tolist(), label='Y='+str(a2)+'*X+'+str(a1))

plt.xlabel('$\\alpha_0$ reference')
plt.ylabel('$\\alpha_0$ predicted')
plt.legend()

#plt.savefig('alpha013_Predicted_GARCH_Gamma6_Linear_MonteCarlo_107_Pena1_01_105106_3.pdf')
plt.show()
plt.close()
