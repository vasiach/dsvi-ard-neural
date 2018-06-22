from sklearn.linear_model import LogisticRegression as LR
import numpy as np
from sklearn.metrics import accuracy_score

# Read the names of the files in lists

train_files = ['data/mnisttxt/train%d.txt' % (i,) for i in range(10)]
test_files = ['data/mnisttxt/test%d.txt' % (i,) for i in range(10)]

# Read train files to arrays, normalize and convert to tensors
tmp = []
for i in train_files:
    with open(i, 'r') as fp:
        tmp += fp.readlines()
train_data = (np.array([[img for img in row.split(" ")] for row in tmp], dtype='int')/255)
print("All train data size: ", train_data.shape)

# Read test files to arrays, normalize and convert to tensors
tmp = []
for i in test_files:
    with open(i, 'r') as fp:
        tmp += fp.readlines()
test_data = np.array([[img for img in row.split(" ")] for row in tmp], dtype='int') / 255

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

tmp = []

# True labels of the test data
for i, file in enumerate(test_files):
    with open(file, 'r') as fp:
        for img in fp:
            tmp.append([1 if j == i else 0 for j in range(0, 10)])
one_hot_test = np.array(tmp)
test_labels = np.array(tmp)
test_labels = np.argmax(test_labels, axis=1)

del tmp[:]
print("Train labels size: ", train_labels.shape)
print("Test labels size: ", test_labels.shape)

lr = LR(penalty='l1', verbose=3, solver='liblinear', max_iter=15000)
lr.fit(train_data, train_labels)
y_pred = lr.predict(test_data)

print('Accuracy of L1-LR {}'.format(accuracy_score(test_labels, y_pred)))

zero_elements = np.sum(lr.coef_ == 0)
sparsity = (zero_elements*1.0)/lr.coef_.size
print('Sparsity of L1-LR {}'.format(sparsity))