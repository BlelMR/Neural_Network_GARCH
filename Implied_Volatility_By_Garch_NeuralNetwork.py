import scipy.stats as si
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt



def Strikefunction(NS, Kmin, Kmax): 
    '''Function for strike plot on range(Kmin, Kmax) with NS strike steps'''
    Strike=np.zeros(NS,'f')
    Strike[0]=Kmin
    for i in range(1,NS):
        #Strike[i]=Strike[i-1]+10
        Strike[i]=Strike[i-1]+(Kmax-Kmin)/NS
        if (Strike[i-1] <100) & (Strike[i]>100):
            Strike[i]=100
        print(Strike[i])
    
    return Strike

def black_call(S, K, T, r, sigma):
    '''Function returning Black call price with zero dividend'''  
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_priceBS = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    #print('call_priceBS=',call_priceBS)
    return call_priceBS

def vegaBS(sigma, K, T, S, r):
    '''Function returning Vega value of a Black Scholes Call''' 
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S*np.sqrt(T)*norm.pdf(d1)

def implied_volatility2(TargetPrice, S, K, T, r, prix_tol = 1.e-5, max_iter=100):
    '''Function returning the implied volatility'''  
    vol = (np.sqrt(2/T * np.abs(np.log(S/K))) +1e-14) #Initialization of the volatility that insures convergence 
    print('Vol0=', vol)
    priceBS =  black_call(S, K, T, r, vol) #initial Call price 
    distance = np.abs(priceBS - TargetPrice)
    iterations = 0
    #Newton algorithm
    while ( (distance > prix_tol) & (iterations < max_iter) ):
        vol = vol - (priceBS-TargetPrice)/vegaBS(vol, K, T, S, r) #Vol Update using the gradient VegaBS
        #priceBS =  BSPut(vol, K, T, S)
        priceBS =  black_call(S, K, T, r, vol)
        distance = np.abs(priceBS - TargetPrice)
        iterations += 1
    return vol
def TargetpriceES(S0, ML, sigma_0, n_steps, alpha0, alpha1, beta1, T, r, Ksig):
    '''Function returning the Call target price using Monte Carlo simulation with the GARCH model and the expectance asset price'''
    TargetPrice=np.zeros(1, 'f')
    #print('Ksig=',Ksig)
    ti=0
    Sigma2_tupdate_0=(sigma_0)**2*np.ones(ML, 'f') #Initialization of sigma**2
    Sommevol2=np.zeros(ML, 'f') #Initialization of Summation term on all sigma**2
    BrTerm=np.zeros(ML, 'f') #Initialization of Brownian Summation term on sigma and Z
    xt2update_0=np.zeros(ML, 'f') #Initialization of the sqaure retunr term  
    Vnull=np.zeros(ML, 'f') # Null vector
    np.random.seed(4) #Initialization of the seed 
    Zn=np.random.normal(0, 1, size=(ML, n_steps)) # Random independent brownian increments on ML trajectories 
    for i in range(n_steps):
        Sommevol2[:]+=Sigma2_tupdate_0*(T/n_steps)
        BrTerm[:]+=np.sqrt(Sigma2_tupdate_0[:])*(Zn[:,i])*(T/n_steps)**(1/2)
        xt2update_0[:]=Sigma2_tupdate_0[:]*(Zn[:,i])**2 #square return value  
        Sigma2_tupdate_0[:]=alpha0+alpha1*(xt2update_0[:])+beta1*(Sigma2_tupdate_0[:]) # GARCH formula 
        #print('Sigma2_tupdate_0=', (Sigma2_tupdate_0))
    ST=S0*np.exp((T-ti)*r-0.5*(Sommevol2[:])+BrTerm[:]) # Final asset price
    #print('Sommevol2=', np.sqrt(Sommevol2))
    #print('exp=',np.exp(-0.5*(Sommevol2[:])))
    print('St=', ST)
    ES=(1/ML)*np.sum(ST[:]) # Excpectance asset price
    print(ES)
    #ES=ES.cpu().numpy()

    TargetPrice=((1/ML)*np.sum(np.maximum(ST[:]-Ksig, Vnull))) # Target price, r=0 !
    print(TargetPrice)
    Ecarttype=np.std(ST[:])
    print('EcartType=', Ecarttype)
    return TargetPrice, ES

def CalculImpliedVOl(Strike, sigma_0, S0, T, n_steps, ML, NS, alpha0, alpha1, beta1):
    '''Function returning the implied volatility array for all strikes'''
    Volim=np.zeros(NS,'f')
    for i in range(NS):
        Ksig=Strike[i]
        #print('Ksig=',Ksig)
        TargetPrice, ES=TargetpriceES(S0, ML, sigma_0, n_steps, alpha0, alpha1, beta1, T, r, Ksig)
        implied_vol = implied_volatility2(TargetPrice, ES, Ksig, T ,r )
        print("Implied volatility:", implied_vol)
        Volim[i]=implied_vol
    return Volim


'''Garch Exact parameters to be filled'''
alpha0=0.000001 #Alpha_0 parameter
alpha1=0.09 #Alpha_1 parameter
beta1=0.8 #Beta_1 parameter

'''Garch Neural Network parameters to be filled'''
alpha0p=0.0011 #Alpha_0 Neural Network parameter
alpha1p=0.091 #Alpha_1 Neural Network parameter
beta1p=0.805 #Beta_1 Neural Network parameter

'''Initialization parameters to be filles'''
sigma_0=0.1 #Garch initial volatility
S0 = 100 #Asset initail price
T = 1/12 #Time Maturity 
n_steps=4 #number of steps for Garch approximation 
r = 0.0 # Interest rate 
q = 0.0 #dividend value
NS=10  #Strike number of steps
ML=1000000 #Monte Carlo Number of trajectories  

Strike=Strikefunction(NS, 70, 140)
print('Sigma Equilibre=', np.sqrt(alpha0/(1-alpha1-beta1))*100 ,'%')


Volimp=np.zeros(NS,'f')
Volimp = CalculImpliedVOl(Strike, sigma_0, S0, T, n_steps, ML, NS, alpha0, alpha1, beta1)
#VolimpP = CalculImpliedVOl(Strike, sigma_0, S0, T, n_steps, ML, NS, alpha0p, alpha1p, beta1p)
#ErrRelativeVolImp=np.sqrt(np.mean(((VolimpP-Volimp)/VolimpP)**2)) # Relative error on the Neural network method
#print('Err Relative VolImp=', ErrRelativeVolImp*100 ,'%')

plt.figure(1)
plt.plot(Strike[:],  Volimp[:]*100, label='Smile Garch Exact')
#plt.plot(Strike[:], VolimpP[:]*100, label='Smile Garch Deep Learning')

plt.xlabel('Strike $K$')
plt.ylabel('Implied Volatility $\sigma$ %')
plt.legend()
#plt.savefig('SmileT10_return_normal.pdf')
plt.show()
