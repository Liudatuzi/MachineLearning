import numpy as np
import random
import matplotlib.pyplot as plt
from math import exp
from sklearn.cluster import KMeans
class Dataset(object):
    def __init__(self,classNum=2,dataNum=200,seed=0,center=None):
        self.classNum=classNum
        self.dataNum=dataNum
        total=classNum*dataNum
        self.data=np.zeros((total,2),dtype=np.float32)
        if center==None:
            self.centers=[(0,0),(2,0),(1,0.9)]
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

class Center(object):
    def __init__(self,x=0,y=0):
        self.x=x
        self.y=y
        self.data=None
        self.count=0


def distance(p1,p2):
    px=p1[0]-p2[0]
    py=p1[1]-p2[1]
    dis=pow(px,2)+pow(py,2)
    return dis


class kmeans(object):
    def __init__(self,nClusters=1,verbose=2):
        self.model=KMeans(
            n_clusters=nClusters,
            verbose=verbose)
        self.nClusters=nClusters
        self.dataset=Dataset(classNum=3)#three class
        self.dataset.generate()
        self.data=self.dataset.data
        self.centers=[]
        self.rho=[]
    def getDensity(self):
        b=0.0
        for i in self.data:
            b+=distance(self.data[0],i)
            #/sum d(x1,xj)
        for i in self.data:
            v=0.0
            #\sum d(xi,xj)/b
            for j in self.data:
                v+=distance(i,j)/b
            self.rho.append(exp(-v))

    def randomCenter(self):
        for i in range(8):
            self.centers.append(Center(x=np.random.normal(1,1),
                                       y=np.random.normal(1,0.6)))
    def show(self):
        plt.figure()
        plt.scatter(self.data[:,0],self.data[:,1],s=10, color='yellow')
        for c in self.centers:
            if c!=None:
                plt.scatter(c.x,c.y, c="blue",label="x", s=25)
        plt.show()

    def RPCL(self):
        self.getDensity()
        self.randomCenter()
        self.show()
        alpha=0.05
        beta=0.05
        while True:
            for index,i in enumerate(self.data):
                dis=[]
                for c in self.centers:#calculate the distance between centers and point i
                    dis.append(distance(i,(c.x,c.y)))

                winId=dis.index(max(dis))
                delta_x=alpha*self.rho[index]*(i[0]-self.centers[winId].x)
                delta_y = alpha * self.rho[index] * (i[1] - self.centers[winId].y)
                self.centers[winId].x+=delta_x
                self.centers[winId].y+=delta_y
                dis[winId]=-1

                rivId=dis.index(max(dis))
                delta_x = alpha * self.rho[index] * (i[0] - self.centers[rivId].x)
                delta_y = alpha * self.rho[index] * (i[1] - self.centers[rivId].y)
                self.centers[rivId].x -= delta_x
                self.centers[rivId].y -= delta_y

            if (delta_x+delta_y)<0.01:
                break

        for i in self.data:
            MIN=1000
            current=None
            for index,c in enumerate(self.centers):
                if distance(i,(c.x,c.y))<MIN:
                    MIN=distance(i,(c.x,c.y))
                    current=index
            self.centers[current].count+=1
        self.show()
        eta=100
        redundant=[]
        for index,c in enumerate(self.centers):
            if c.count<eta:
                redundant.append(index)
        for i in redundant:
            self.centers[i]=None
        self.show()






if __name__ =='__main__':
    kmeans=kmeans()
    kmeans.RPCL()












