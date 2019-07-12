import numpy as np
from numpy import *
import math
from sklearn.decomposition import PCA, FactorAnalysis
import matplotlib as mpl
import matplotlib.pyplot as plt

def Mu(m,mu):
    Mu=[]
    for i in range(m):
        Mu.append(mu)
    return Mu
def I(m,Sigma):
    I=np.eye(m,dtype=int)
    I=I*Sigma
    return I

def dataset(N,n,m,Sigma,mu):
    Mu_y=Mu(m,mu)
    Mu_e=Mu(n,mu)
    y = np.random.multivariate_normal(mean=Mu_y, cov=I(m,1), size = n)
    y=mat(y)
    A=mat(np.random.random((N,m)))
    E=mat(np.random.multivariate_normal(mean=Mu_e, cov=I(n,Sigma=Sigma), size = N))
    mm=A*y.T+E
    mm=array(mm)
    return mm
def aic(mm):
    aic=[]
    for i in range(1,10):
        fa=FactorAnalysis(n_components=i,tol=0.0001,max_iter=5000)
        fa.fit(mm)
        d=n*i
        b=100*fa.score(mm)-d
        aic.append(b)
    return aic

def bic(mm):
    bic=[]
    for i in range(1,10):
        fa=FactorAnalysis(n_components=i,tol=0.0001,max_iter=5000)
        fa.fit(mm)
        d=n*i
        b=100*fa.score(mm)-(math.log(100)*d)/2
        bic.append(b)
    return bic
def show(aid,bic):
    x = [i for i in range(1, 10)]
    plt.figure()
    plt.plot(x, aic, label='aic')
    plt.plot(x, bic, label='bic')
    plt.xlabel('n_components')
    plt.ylabel('score')
    plt.legend()
    # plt.savefig('n_components_3')
    plt.show()
if __name__ == '__main__':
    N = 100
    n = 10
    m = 3
    Sigma = 0.1
    mu = 0
    mm=dataset(N,n,m,Sigma,mu)
    aic=aic(mm)
    bic=bic(mm)
    show(aic,bic)