import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


def load_data():

    # Read the names of the files in lists

    train_files = ['data/mnisttxt/train%d.txt' % (i,) for i in range(10)]
    test_files = ['data/mnisttxt/test%d.txt' % (i,) for i in range(10)]

    # Read train files to arrays, normalize and convert to tensors
    tmp = []
    for i in train_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    train_data = Variable(torch.FloatTensor(np.array([[img for img in row.split(" ")] for row in tmp], dtype='int')/255))
    print("All train data size: ", train_data.shape)

    # Read test files to arrays, normalize and convert to tensors
    tmp = []
    for i in test_files:
        with open(i, 'r') as fp:
            tmp += fp.readlines()
    test_data = Variable(torch.FloatTensor(
                        np.array([[img for img in row.split(" ")] for row in tmp], dtype='int') / 255))

    print("Test data size: ", test_data.shape)

    # True labels of the train data
    tmp = []
    for i, file in enumerate(train_files):
        with open(file, 'r') as fp:
            for img in fp:
                tmp.append([1 if j == i else 0 for j in range(0, 10)])

    one_hot_train = np.array(tmp)
    train_labels = np.array(tmp)
    train_labels = np.argmax(train_labels, axis=1)
    train_labels = Variable(torch.LongTensor(train_labels))

    tmp = []

    # True labels of the test data
    for i, file in enumerate(test_files):
        with open(file, 'r') as fp:
            for img in fp:
                tmp.append([1 if j == i else 0 for j in range(0, 10)])
    one_hot_test = np.array(tmp)
    test_labels = np.array(tmp)
    test_labels = np.argmax(test_labels, axis=1)
    test_labels = Variable(torch.LongTensor(test_labels))

    del tmp[:]
    print("Train labels size: ", train_labels.shape)
    print("Test labels size: ", test_labels.shape)
    return train_data, test_data, train_labels, test_labels, one_hot_train, one_hot_test


def choose_class(x_tr, y_tr, digit):

    y_tr = y_tr.data.numpy()
    x_tr = x_tr.data.numpy()

    tmp = []
    tmp_lbl = []

    for i in range(len(x_tr)):
        if y_tr[i] == digit:
            tmp.append(x_tr[i])
            tmp_lbl.append(y_tr[i])

    return tmp, tmp_lbl


def binary_load_data():

    # Load data. Labels are returned both as one hot vectors and as vectors with the digits.

    x_tr, x_tst, y_tr, y_tst, y_one_train, y_one_test = load_data()

    # Choose digit as class 0 for training
    x_tr_zeros, y_tr_zeros,  = choose_class(x_tr, y_tr, 2)
    y_tr_zeros = np.zeros((len(y_tr_zeros), 1))

    # Choose digit as class 1 for training
    x_tr_ones, y_tr_ones = choose_class(x_tr, y_tr, 5)
    y_tr_ones = np.ones((len(y_tr_ones), 1))

    # Concatenate, shuffle and break data to X_train and labels
    x_btr = np.array(np.concatenate((x_tr_zeros, x_tr_ones), axis=0))
    y_btr = np.array(np.concatenate((y_tr_zeros, y_tr_ones), axis=0))

    alldata = np.column_stack((x_btr, y_btr))

    np.random.shuffle(alldata)

    y_tr = alldata[:, -1]
    x_tr = np.delete(alldata, -1, axis=1)

    # Convert arrays to torch Variables
    x_tr = Variable(torch.FloatTensor(x_tr))
    y_tr = Variable(torch.FloatTensor(y_tr))

    # Choose digit as class 0 for test
    x_tst_zeros, y_tst_zeros = choose_class(x_tst, y_tst, 2)
    y_tst_zeros = np.zeros((len(y_tst_zeros), 1))

    # Choose digit as class 1 for test
    x_tst_ones, y_tst_ones = choose_class(x_tst, y_tst, 5)
    y_tst_ones = np.ones((len(y_tst_ones), 1))

    # Concatenate, permutate and break data to X_test and labels

    x_btst = np.concatenate((x_tst_zeros, x_tst_ones), axis=0)
    y_btst = np.concatenate((y_tst_zeros, y_tst_ones), axis=0)

    alldata_tst = np.column_stack((x_btst, y_btst))
    np.random.shuffle(alldata_tst)

    y_tst = alldata_tst[:, -1]
    x_tst = np.delete(alldata_tst, -1, axis=1)

    # Convert arrays to torch Variables
    x_tst = Variable(torch.FloatTensor(x_tst))
    y_tst = Variable(torch.FloatTensor(y_tst))

    print('Train data size : {}'.format(x_tr.shape))
    print('Test data size : {}'.format(x_tst.shape))

    return x_tr, y_tr, x_tst, y_tst

def img_show():

    x_tr, y_tr, x_tst, y_tst= binary_load_data()
    pixels = x_tst[779,:]
    pixels = pixels.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

