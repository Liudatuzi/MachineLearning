import numpy as np
def unpickle(file):#CIFAR-10官方给出的使用方法
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='iso-8859-1')
    return dict


for i in range(1, 6):
    file = 'data_batch_%s' % i
    dict_train_batch = unpickle(file)  # 将data_batch文件读入到数据结构(字典)中
    data_train_batch = dict_train_batch.get('data')  # 字典中取data
    labels = dict_train_batch.get('labels')  # 字典中取labels
    data_name = file + '_data.npy'
    np.save(data_name, data_train_batch)
    label_name = file + '_label.npy'
    np.save(label_name, labels)

dict_test_batch = unpickle("test_batch")  # 将data_batch文件读入到数据结构(字典)中
data_test_batch = dict_test_batch.get('data')  # 字典中取data
labels = dict_test_batch.get('labels')  # 字典中取labels
np.save("test_batch_data.npy",data_test_batch)
np.save("test_batch_label.npy",labels)
#concatenate labels
a=np.load("data_batch_1_label.npy")
for i in range(2, 6):
    tmp = np.load("data_batch_%s_label.npy" % i)
    a = np.hstack((a, tmp))
np.save("total_label.npy",a)

#concatenate data
a=np.load("data_batch_1_data.npy")
for i in range(2, 6):
    tmp = np.load("data_batch_%s_data.npy" % i)
    a = np.vstack((a, tmp))
np.save("total_data.npy",a)