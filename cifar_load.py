import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def visualize_image(X, Y, names, id):
    rgb = X[id, :]
    img = rgb.reshape(3, 32, 32).astype('uint8')
    plt.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')
    plt.title(names[id])
    plt.show()


def load_data():
    data_batch1 = unpickle('data/cifar/data_batch_1')
    data_batch2 = unpickle('data/cifar/data_batch_2')
    data_batch3 = unpickle('data/cifar/data_batch_3')
    data_batch4 = unpickle('data/cifar/data_batch_4')
    data_batch5 = unpickle('data/cifar/data_batch_5')

    X_train = np.concatenate((data_batch1['data'], data_batch2['data'],
                              data_batch3['data'], data_batch4['data'], data_batch5['data']), axis=0)

    X_train = Variable(torch.FloatTensor(X_train/255))

    print("All train data size: ", X_train.shape)
    labels_1 = np.array(data_batch1['labels'])
    labels_2 = np.array(data_batch2['labels'])
    labels_3 = np.array(data_batch3['labels'])
    labels_4 = np.array(data_batch4['labels'])
    labels_5 = np.array(data_batch5['labels'])

    T = np.array(np.concatenate((labels_1, labels_2, labels_3, labels_4, labels_5), axis=0)).astype('int')

    T_one = np.zeros((T.shape[0], 10))
    for i in range(T.shape[0]):
        T_one[i, T[i]] = 1

    names = data_batch1['filenames']

    test_data = unpickle('data/cifar/test_batch')

    X_test = Variable(torch.FloatTensor(test_data['data']/255))
    print("Test data size: ", X_test.shape)

    T_test = np.array(test_data['labels'])

    T_one_test = np.zeros((T_test.shape[0], 10))
    for i in range(T_test.shape[0]):
        T_one_test[i, T_test[i]] = 1

    T = Variable(torch.LongTensor(T))
    T_test = Variable(torch.LongTensor(T_test))
    print("Train labels size: ", T.shape)
    print("Test labels size: ", T_test.shape)

    return X_train, X_test, T, T_test, T_one, T_one_test

