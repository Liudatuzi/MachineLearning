import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
train_data=np.load("G:/cs231/assignment2/cs231n/datasets/cifar-10-batches-py/total_train.npy")
train_label=np.load("G:/cs231/assignment2/cs231n/datasets/cifar-10-batches-py/total_label.npy")
test_data=np.load("G:/cs231/assignment2/cs231n/datasets/cifar-10-batches-py/test_batch_data.npy")
test_label=np.load("G:/cs231/assignment2/cs231n/datasets/cifar-10-batches-py/test_batch_label.npy")
def original(train_data,train_label,test_data,test_label):
    model = SVC(kernel='rbf', gamma=1e-4, C=10)
    model.fit(train_data, train_label)
    print(model.score(test_data, test_label))
def scaler(train_data,train_label,test_data,test_label):
    ss = StandardScaler()
    train_data = ss.fit_transform(train_data)
    test_data = ss.fit_transform(test_data)
    model = SVC(kernel='rbf', gamma=1e-4, C=10)
    model.fit(train_data, train_label)
    print(model.score(test_data, test_label))


def pca_scale(train_data,train_label,test_data,test_label):
    ss = StandardScaler()
    train_data = ss.fit_transform(train_data)
    test_data = ss.fit_transform(test_data)
    model = SVC(kernel='rbf', gamma=1e-4, C=10)
    for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
        num = int(3072 * i)
        pca = PCA(n_components=num)
        new_train = deepcopy(train_data)
        new_test = deepcopy(test_data)
        new_train = pca.fit_transform(new_train)
        new_test = pca.fit_transform(new_test)
        model.fit(new_train, train_label)
        print(i, ' ', model.score(new_test, test_label))
