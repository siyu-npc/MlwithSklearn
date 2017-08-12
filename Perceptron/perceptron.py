import numpy as np
from sklearn.linear_model import Perceptron
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification

#line:2*x1 + x2 - 10 = 0
#generate the positive sample

x,y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1)
x_min,x_max = np.floor(x[:,0].min()),np.ceil(x[:,0].max())
# print(y)
#positive = []
#x1 = np.random.randint(-20,20,(1000,))
#print(x1)
#tmp = np.random.randn(1000) * 20
#x2 = -2 * x1 + 10 + tmp
#print(x2)
#target = []
#for i in range(1000):
#    if x2[i] >= 2.0:
#        x2[i] += 5.
#        target.append(1)
#    else:
#        x2[i] -= 5.
#        target.append(-1)
#print(target)
positive_x1 = [x[i,0] for i in range(1000) if y[i] == 1]
positive_x2 = [x[i,1] for i in range(1000) if y[i] == 1]
negetive_x1 = [x[i,0] for i in range(1000) if y[i] == 0]
negetive_x2 = [x[i,1] for i in range(1000) if y[i] == 0]

#x_data = np.zeros(shape=(1000,2))
#for i in range(1000):
#    x_data[i,0] = x1[i]
#    x_data[i,1] = x2[i]
#y_data = np.array(target)
#print(x_data.shape)
#x_data = np.rot90(x_data)
#y_data = np.rot90(y_data)
x_data_train = x[:800,:]
x_data_test = x[800:,:]
y_data_train = y[:800]
y_data_test = y[800:]
#regr = linear_model.LinearRegression()
#regr.fit(x_data_train,y_data_train)
#print(regr.coef_)
#print(regr.score(x_data_train,y_data_train))
#score = regr.score(x_data_test,y_data_test)
#print(score)
#plt.scatter(positive_x,positive_y)
#plt.scatter(negetive_x,negetive_y)
#y = x1 * regr.coef_[0] + regr.coef_[1]
#plt.plot(x1,y)
#plt.show()
clf = Perceptron(fit_intercept=False,n_iter=30,shuffle=False).fit(x_data_train,y_data_train)
print(clf.coef_)
print(clf.score(x_data_test,y_data_test))
#print(clf.predict(x_data_test))
#print(y_data_test)
print(clf.intercept_)
plt.scatter(positive_x1,positive_x2,c='red')
plt.scatter(negetive_x1,negetive_x2,c='blue')
line_x = np.arange(x_min,x_max)
line_y = line_x * (-clf.coef_[0][0] / clf.coef_[0][1]) - clf.intercept_
plt.plot(line_x,line_y)
plt.show()