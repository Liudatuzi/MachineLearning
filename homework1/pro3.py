import numpy as np
import random
import matplotlib.pyplot as plt
from math import exp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from copy import deepcopy

from sklearn.mixture import BayesianGaussianMixture
class Dataset(object):
    def __init__(self,classNum=2,dataNum=200,seed=0,center=None):
        self.classNum=classNum
        self.dataNum=dataNum
        total=classNum*dataNum
        self.data=np.zeros((total,2),dtype=np.float32)
        if center==None:
            self.centers=[(0,0),(2,0),(1,1),(2,2),(0,2)]
        else:
            self.centers=center
        self.colors=['yellow','pink','blue','yellow']
        random.seed(seed)

    def generate(self):
        for k in range(self.classNum):#the kth class and center
            kcx=self.centers[k][0]
            kcy=self.centers[k][1]
            sigma=random.random()*0.4
            #produce the point for the kth class
            for i in range(self.dataNum):
                self.data[k*self.dataNum+i][0]=np.random.normal(kcx,sigma)
                self.data[k*self.dataNum+i][1]=np.random.normal(kcy,sigma)

    def shown(self):
        plt.figure()
        for i in range(self.classNum):
            color=self.colors[i%4]
            for j in range(self.dataNum):
                x=self.data[i*self.dataNum+j][0]
                y=self.data[i*self.dataNum+j][1]
                plt.scatter(x,y,s=8,c=color)
        plt.show()

class VBEM(object):
    def __init__(self,n_components=1,verbose=2,verbose_interval=1,data=None):
        self.model=BayesianGaussianMixture(
            n_components=n_components,
            verbose=verbose,
            verbose_interval=verbose_interval
        )
        self.n_components=n_components
        if data==None:
            self.dataset=Dataset()
            self.dataset.generate()
        else:
            self.dataset=data
        self.data=self.dataset.data
    def train(self):
        self.model.fit(self.data)
    def show(self,n=None):
        plt.figure()
        self.model.fit(self.data)
        labels=self.model.predict(self.data)
        plt.scatter(self.data[:,0],self.data[:,1],c=labels,s=10)
        if n==None:
            plt.show()
        else:
            plt.savefig('Pro2/vbem_%d_%d'%(n,4))

class GMM_EM(object):
    def __init__(self,n_components=1,verbose=2,verbose_interval=1,data=None):
        self.model=GaussianMixture(
            n_components=n_components,
            verbose=verbose,
            verbose_interval=verbose_interval
        )
        self.n_components=n_components
        if data==None:
            self.dataset=Dataset()
            self.dataset.generate()
        else:
            self.dataset=data
        self.data=self.dataset.data
        self.aic=[]
        self.bic=[]
        self.aic_b=None


    def train(self):
        self.model.fit(self.data)
    def aic_select(self):
        self.aic_b=True
        minaic=9999
        for n in range(1,self.n_components+1):
            gmm=GaussianMixture(n_components=n)
            gmm.fit(self.data)
            self.aic.append(gmm.aic(self.data))
            if self.aic[-1]<minaic:
                minaic=self.aic[-1]
                self.model=deepcopy(gmm)
        print("aic\n",self.aic)
        self.res_n=self.aic.index(minaic)+1
        print("selected components:",self.res_n,'\n')
    def bic_select(self):
        self.aic_b=False
        minbic=9999
        for n in range(1,self.n_components+1):
            gmm=GaussianMixture(n_components=n)
            gmm.fit(self.data)
            self.bic.append(gmm.bic(self.data))
            if self.bic[-1]<minbic:
                minbic=self.bic[-1]
                self.model=deepcopy(gmm)
        print("bic\n",self.bic)
        self.res_n=self.bic.index(minbic)+1
        print("selected components:",self.res_n,'\n')

    def show(self,n=None):
        plt.figure()
        labels=self.model.predict(self.data)
        plt.scatter(self.data[:,0],self.data[:,1],c=labels,s=15)
        if n==None:
            plt.show()
        else:
            if self.aic_b:
                plt.savefig('Pro2/aic_%d_%d'%(n,self.res_n))
            else:
                plt.savefig('Pro2/bic_%d_%d' % (n, self.res_n))

def testGMM(data,n,aic=True):
    model=GMM_EM(n_components=10,data=data)
    if aic:
        print("test GMM selecting AIC\n")
        model.aic_select()
    else:
        print("test GMM selecting BIC\n")
        model.bic_select()
    model.show(n)
def testVBEM(data,n):
    print("test VBEM\n")
    model=VBEM(n_components=10,data=data)
    model.show(n)


def sample_size():
    samples=[10,20,40,80,160]
    for i in samples:
        data=Dataset(classNum=5,dataNum=i)
        data.generate()
        testGMM(data,i)#test aic
        testGMM(data,i,False)#test bic
        testVBEM(data,i)



def cluster_distance():


    dis=[0.1,0.4,0.7,1.1,3.0]
    centers = [[(0, 0), (0,dis[0]*2),(dis[0]*2, 0), (dis[0]*2,dis[0]*2),(dis[0],dis[0])],
               [(0, 0), (0, dis[1] * 2), (dis[1] * 2, 0), (dis[1] * 2, dis[1] * 2), (dis[1], dis[1])],
               [(0, 0), (0, dis[2] * 2), (dis[2] * 2, 0), (dis[2] * 2, dis[2] * 2), (dis[2], dis[2])],
               [(0, 0), (0, dis[3] * 2), (dis[3] * 2, 0), (dis[3] * 2, dis[3] * 2), (dis[3], dis[3])],
               [(0, 0), (0, dis[4] * 2), (dis[4] * 2, 0), (dis[4] * 2, dis[4] * 2), (dis[4], dis[4])],

               ]

    for i, center in enumerate(centers):
        data = Dataset(classNum=5, dataNum=50, center=center)
        data.generate()
        testGMM(data, int(dis[i] * 10))
        testGMM(data, int(dis[i] * 10), False)
        testVBEM(data, int(dis[i] * 10))




if __name__ == '__main__':

    cluster_distance()