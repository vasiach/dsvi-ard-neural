import numpy as np
from cifar_load import load_data
from model_ard import dsviard
import matplotlib.pyplot as plt
from matplotlib.colors import cnames as mcolors
from softmax import softmax_n

X, X_test, T, T_test, T_one_train, T_one_test = load_data()

T_one_train = T_one_train.astype(float)
T_one_test = T_one_test.astype(float)

num_classes = 10
D, D_out = X.shape[1], num_classes


F, mu_in, C_in, mu_out, C_out = dsviard(X, T, D, D_out, reg_flag=False)

mF = np.zeros((len(F), 1))
W = 200

for i in range(1, len(F)):
    st = i-W+1
    if st < 1:
        st = 1
    mF[i] = np.mean(F[st:i])

Cov = C_in.numpy()
vars = np.multiply(Cov, Cov)
meandist = mu_in.numpy()
meandist = meandist.T


# plot lower bound mean
plt.plot(mF, 'b', linewidth=1)
plt.xlabel('iterations')
plt.ylabel('lower bound')
plt.gca()
plt.show()

# plot lower bound
plt.plot(F, 'b', linewidth=1)
plt.xlabel('iterations')
plt.ylabel('lower bound')
plt.gca()
plt.show()


# sparsity of mean var dist
colors = list(mcolors.keys())
for i in range(D_out):
    plt.plot(range(D), meandist[:, i], '.', color=colors[i], markersize=3)
plt.xlabel('variable index')
plt.ylabel('mean of var dist')
rnge = np.max(meandist)-np.min(meandist)
plt.axis([0, len(meandist)+1, (np.min(meandist)-0.02*rnge), np.max(meandist+0.02*rnge)])
plt.show()

mu_in = mu_in.numpy()
mu_out = mu_out.numpy()
X = X.data.numpy()
X_test = X_test.data.numpy()
T = T.data.numpy()
T_test = T_test.data.numpy()


H = np.tanh(X.dot(mu_in.T))
Str = softmax_n(H.dot(mu_out.T))

H_test = np.tanh(X_test.dot(mu_in.T))
Sts = softmax_n(H_test.dot(mu_out.T))


T = T.reshape(T.shape[0], 1)
T_test = T_test.reshape(T_test.shape[0], 1)
train_error = np.sum(np.abs(T_one_train-np.around(Str)))
test_error = np.sum(np.abs(T_one_test-np.around(Sts)))

print('MNIST: Train error {}/{}'.format(train_error, len(X)))
print('MNIST: Test error {}/{}'.format(test_error, len(X_test)))

sparsity = (np.count_nonzero(meandist)/meandist.size)*100

print('Sparsity(non-zero/size) {}'.format(sparsity))


image_test = meandist[:, 3]
image_test = image_test*255


img = image_test.reshape(3, 32, 32).astype('uint8')
plt.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')
plt.show()

