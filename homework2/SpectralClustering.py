from sklearn import datasets
from matplotlib import pyplot as plt
from itertools import cycle, islice
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

from sklearn.datasets import make_moons
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_classification
def genTwoCircles(n_samples=1000):
    X,y = datasets.make_circles(n_samples, factor=0.5, noise=0.05)
    return X, y


def plot(x, y,y_pred, k_pred,g_pred):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#999999', '#e41a1c', '#dede00']),
                                        int(max(y_pred) + 1))))

    plt.subplot(141)
    plt.scatter(x[:, 0], x[:, 1], s=10, color=colors[y])
    plt.title("Original")
    plt.subplot(142)
    plt.scatter(x[:, 0], x[:, 1], s=10, color=colors[y_pred])
    plt.title("Spectral Clustering")
    plt.subplot(143)
    plt.scatter(x[:, 0], x[:, 1], s=10, color=colors[k_pred])
    plt.title("Kmeans Clustering")
    plt.subplot(144)
    plt.scatter(x[:, 0], x[:, 1], s=10, color=colors[g_pred])
    plt.title("Gaussian Mixture")
    plt.show()
    #plt.savefig("../figures/spectral_clustering.png")

def circle():
    plt.rcParams['figure.figsize'] = (18.0,4.0)
    x,y=genTwoCircles(1000)
    y_pred = SpectralClustering(affinity='rbf',n_clusters=2, n_neighbors=10).fit_predict(x)
    k_pred=KMeans(n_clusters=2).fit_predict(x)
    g_pred = GaussianMixture(n_components=2, covariance_type='diag', random_state=0).fit_predict(x)
    plot(x,y,y_pred,k_pred,g_pred)
def moon():
    plt.rcParams['figure.figsize'] = (18.0,4.0)
    x,y=make_moons(n_samples=1000,noise=0.1)
    y_pred = SpectralClustering(affinity='nearest_neighbors',n_clusters=2, n_neighbors=10).fit_predict(x)
    k_pred=KMeans(n_clusters=2).fit_predict(x)
    g_pred = GaussianMixture(n_components=2, covariance_type='diag', random_state=0).fit_predict(x)
    plot(x,y,y_pred,k_pred,g_pred)
def ordinary():
    plt.rcParams['figure.figsize'] = (18.0, 4.0)
    center = [[1, 1], [-1, -1], [1, -1]]
    x, y = make_blobs(n_samples=1000, centers=center, n_features=2,
                           cluster_std=0.3, random_state=0)
    y_pred = SpectralClustering(affinity='nearest_neighbors',n_clusters=3, n_neighbors=10).fit_predict(x)
    k_pred=KMeans(n_clusters=3).fit_predict(x)
    g_pred = GaussianMixture(n_components=3, covariance_type='diag', random_state=0).fit_predict(x)
    plot(x, y, y_pred, k_pred, g_pred)
def classify():
    plt.rcParams['figure.figsize'] = (18.0, 4.0)
    x,y = make_classification(n_samples=500, n_features=3, n_redundant=0, n_informative=2,
                                    random_state=1, n_clusters_per_class=2)
    y_pred = SpectralClustering(affinity='nearest_neighbors', n_clusters=2, n_neighbors=10).fit_predict(x)
    k_pred = KMeans(n_clusters=2).fit_predict(x)
    g_pred=GaussianMixture(n_components=2, covariance_type='diag', random_state=0).fit_predict(x)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(y_pred) + 1))))
    plt.subplot(141)
    plt.scatter(x[:, 0], x[:, 1], s=10, color=colors[y])
    plt.title("Original")
    plt.subplot(142)
    plt.scatter(x[:, 0], x[:, 1], s=10, color=colors[y_pred])
    plt.title("Spectral Clustering")
    plt.subplot(143)
    plt.scatter(x[:, 0], x[:, 1], s=10, color=colors[k_pred])
    plt.title("Kmeans Clustering")
    plt.subplot(144)
    plt.scatter(x[:, 0], x[:, 1], s=10, color=colors[g_pred])
    plt.title("Gaussian Mixture")
    plt.show()



def no_structure():
    plt.rcParams['figure.figsize'] = (18.0, 4.0)
    no_structure = np.random.rand(1000, 2), None
    no_structure = np.array(no_structure)
    x=no_structure[0]
    y=[]
    for i in range(len(x)):
        y.append(1)
    y_pred = SpectralClustering(affinity='nearest_neighbors', n_clusters=3, n_neighbors=10).fit_predict(x)
    k_pred = KMeans(n_clusters=3).fit_predict(x)
    g_pred = GaussianMixture(n_components=3, covariance_type='diag', random_state=0).fit_predict(x)
    plot(x, y, y_pred, k_pred, g_pred)
no_structure()

# print(y_pred)
# print(k_pred)
#plt.figure()
#plt.scatter(x[:,0],x[:,1])
#plt.show()