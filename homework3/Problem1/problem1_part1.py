import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
x1 = np.load('data/diabetes/data.npy')
y1 = np.load('data/diabetes/label.npy')
x2 = np.load('data/breast_cancer/data.npy')
y2 = np.load('data/breast_cancer/label.npy')
def test_data_size(x,y):
    test_size=[0.1,0.2,0.3,0.4,0.5]
    svm_result=[]
    mlp_result=[]
    for size in test_size:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)
        svm=SVC()
        mlp=MLPClassifier()
        svm.fit(x_train,y_train)
        mlp.fit(x_train,y_train)
        svm_result.append(svm.score(x_test,y_test))
        mlp_result.append(mlp.score(x_test,y_test))
    return test_size,svm_result,mlp_result
def draw_size():
    size, svm, mlp = test_data_size(x1,y1)
    plt.figure()
    plt.plot(size, svm, label='SVM', marker='o')
    plt.plot(size, mlp, label='MLP', marker='p')
    max_index1 = svm.index(max(svm))
    max_index2 = mlp.index(max(mlp))
    plt.text(size[max_index1], svm[max_index1], round(svm[max_index1], 3))
    plt.text(size[max_index2], mlp[max_index2], round(mlp[max_index2], 3))
    plt.xlabel('test_size')
    plt.ylabel('score')
    plt.xticks(size)
    plt.legend()
    plt.title("breast_cancer")
    plt.show()

def svm_para(trainData,trainLabel):
    param_test = {'C': [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 3.0, 10.0,100.0,1000.0],
                  'gamma': [1e-3, 1e-4, 'auto'],
                  'kernel':[ 'linear', 'rbf', 'poly']}
    gs1 = GridSearchCV(SVC(), param_grid=param_test, cv=3, scoring='r2')
    gs1.fit(trainData, trainLabel)
    print(gs1.best_score_)
    print(gs1.best_params_)
def mlp_para(trainData,trainLabel):
    param_test={'solver':['lbfgs','sgd','adam'],
                'hidden_layer_sizes':[(100,), (10, 20), (200, 200), (100, 100, 100)],
                'alpha':[1e-3, 1e-4],
                'activation':['identity', 'logistic', 'tanh', 'relu'],
                'learning_rate':['constant','invscaling','adaptive']}
    gs2=GridSearchCV(MLPClassifier(max_iter=1000),param_grid=param_test,cv=3,scoring='r2')
    gs2.fit(trainData,trainLabel)
    print(gs2.best_score_)
    print(gs2.best_params_)
def Compare_C(x1,y1,x2,y2):
    diabetes = []
    breast = []
    for C in [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 3.0, 10.0, 100.0, 1000.0]:
        model = SVC(kernel='rbf', gamma=1e-4, C=C)
        x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2)
        model.fit(x_train, y_train)
        t = model.score(x_test, y_test)
        diabetes.append(t)
        print(t)
    for C in [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 3.0, 10.0, 100.0, 1000.0]:
        model = SVC(kernel='rbf', gamma=1e-4, C=C)
        x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size=0.2)
        model.fit(x_train, y_train)
        t = model.score(x_test, y_test)
        breast.append(t)
        print(t)

    plt.figure()
    a = diabetes.index(max(diabetes))
    b = breast.index(max(breast))
    x = [i for i in range(len(diabetes))]
    CC = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 10, 100, 1000]
    plt.plot(diabetes, label='diabetes')
    plt.plot(breast, label='breast_cancer')
    plt.text(x[a], diabetes[a], round(diabetes[a], 3))
    plt.text(x[b], breast[b], round(breast[b], 3))
    plt.xlabel('C')
    plt.ylabel('score')
    plt.xticks(x, CC)
    plt.legend()
    plt.show()
def com_layer(x1,y1,x2,y2):
    diabetes = []
    breast = []
    for layer in [(100,), (10, 20), (100, 100), (200, 200), (100, 200, 100)]:
        model = MLPClassifier(solver='adam', alpha=1e-4, activation='relu', learning_rate='invscaling',
                              hidden_layer_sizes=layer)
        x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2)
        model.fit(x_train, y_train)
        t = model.score(x_test, y_test)
        diabetes.append(t)
        print(t)
    for layer in [(100,), (10, 20), (100, 100), (200, 200), (100, 200, 100)]:
        model = MLPClassifier(solver='adam', alpha=1e-3, activation='relu', learning_rate='invscaling',
                              hidden_layer_sizes=layer)
        x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size=0.2)
        model.fit(x_train, y_train)
        t = model.score(x_test, y_test)
        breast.append(t)
        print(t)

    plt.figure()
    a = diabetes.index(max(diabetes))
    b = breast.index(max(breast))
    x = [i for i in range(len(diabetes))]
    CC = [(100,), (10, 20), (100, 100), (200, 200), (100, 200, 100)]
    plt.plot(diabetes, label='diabetes')
    plt.plot(breast, label='breast_cancer')
    plt.text(x[a], diabetes[a], round(diabetes[a], 3))
    plt.text(x[b], breast[b], round(breast[b], 3))
    plt.xlabel('layer')
    plt.ylabel('score')
    plt.xticks(x, CC)
    plt.legend()
    plt.show()
def data_dimension(x1,y1,x2,y2):
    model1 = SVC(kernel='rbf', gamma=1e-4, C=10)
    model2 = MLPClassifier(solver='adam', alpha=1e-3, activation='relu', learning_rate='invscaling',
                           hidden_layer_sizes=(200, 200))
    # for diabetes 8 features
    t1 = []
    t2 = []
    for i in range(1, 9):
        pca = PCA(n_components=i)
        new_x1 = pca.fit_transform(x1)
        x1_train, x1_test, y1_train, y1_test = train_test_split(new_x1, y1, test_size=0.4)
        model1.fit(x1_train, y1_train)
        t1.append(model1.score(x1_test, y1_test))
        model2.fit(x1_train, y1_train)
        t2.append(model2.score(x1_test, y1_test))
    t11 = []
    t22 = []
    for i in range(1, 11):
        pca = PCA(n_components=i)
        new_x2 = pca.fit_transform(x2)
        x2_train, x2_test, y2_train, y2_test = train_test_split(new_x2, y2, test_size=0.4)
        model1.fit(x2_train, y2_train)
        t11.append(model1.score(x2_test, y2_test))
        model2.fit(x2_train, y2_train)
        t22.append(model2.score(x2_test, y2_test))
    return t1,t2,t11,t22
def data_scale(x1,y1,x2,y2):
    model1 = SVC(kernel='rbf', gamma=1e-4, C=10)
    model2 = MLPClassifier(solver='adam', alpha=1e-3, activation='relu', learning_rate='invscaling',
                           hidden_layer_sizes=(200, 200))
    ss = StandardScaler()
    new_x1 = ss.fit_transform(x1)
    new_x2 = ss.fit_transform(x2)
    x1_train, x1_test = train_test_split(x1, test_size=0.4)
    x2_train, x2_test = train_test_split(x2, test_size=0.4)
    nx1_train, nx1_test, y1_train, y1_test = train_test_split(new_x1, y1, test_size=0.4)
    nx2_train, nx2_test, y2_train, y2_test = train_test_split(new_x2, y2, test_size=0.4)

    model1.fit(x1_train, y1_train)
    print(model1.score(x1_test, y1_test))
    model1.fit(nx1_train, y1_train)
    print(model1.score(nx1_test, y1_test))
    model1.fit(x2_train, y2_train)
    print(model1.score(x2_test, y2_test))
    model1.fit(nx2_train, y2_train)
    print(model1.score(nx2_test, y2_test))

    model2.fit(x1_train, y1_train)
    print(model2.score(x1_test, y1_test))
    model2.fit(nx1_train, y1_train)
    print(model2.score(nx1_test, y1_test))
    model2.fit(x2_train, y2_train)
    print(model2.score(x2_test, y2_test))
    model2.fit(nx2_train, y2_train)
    print(model2.score(nx2_test, y2_test))
# svm_para(x1,y1)