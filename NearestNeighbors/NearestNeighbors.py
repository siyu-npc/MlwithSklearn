import sklearn.neighbors as neighbors
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

x,y = make_classification(n_samples=1000,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1,n_classes=3)
x_min,x_max = x[:,0].min() - 1,x[:,0].max() + 1
y_min,y_max = x[:,1].min() - 1,x[:,1].max() + 1
# print(y)
# print(type(y))
sample_class1 = np.array([x[i] for i in range(1000) if y[i] == 0])
sample_class2 = np.array([x[i] for i in range(1000) if y[i] == 1])
sample_class3 = np.array([x[i] for i in range(1000) if y[i] == 2])

plt.scatter(sample_class1[:,0],sample_class1[:,1],c='red')
plt.scatter(sample_class2[:,0],sample_class2[:,1],c='blue')
plt.scatter(sample_class3[:,0],sample_class3[:,1],c='green')
#plt.show()

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

x_train = x[:800]
y_train = y[:800]
x_test = x[800:]
y_test = y[800:]

# step_size = 0.02
# for weights in ['uniform','distance']:
#     clf = neighbors.KNeighborsClassifier(n_neighbors=10,weights=weights)
#     clf.fit(x_train,y_train)
#     # print(clf.score(x_test,y_test))
#     xx,yy = np.meshgrid(np.arange(x_min,x_max,step_size),np.arange(y_min,y_max,step_size))
#     z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
#     print(xx.shape)
#     print(z.shape)
#     # z = clf.predict(x_test)
#     z = z.reshape(xx.shape)
#     plt.figure()
#     plt.pcolormesh(xx,yy,z,cmap=cmap_light)
#     # plt.scatter(x[:,0],x[:,1],c = y,cmap=cmap_bold)
#     plt.xlim(xx.min(),xx.max())
#     plt.ylim(yy.min(),yy.max())
clf = neighbors.KNeighborsClassifier(n_neighbors=10,weights='uniform')
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))

step_size = 0.02
xx,yy = np.meshgrid(np.arange(x_min,x_max,step_size),np.arange(y_min,y_max,step_size))
z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,z,cmap=cmap_light)
plt.scatter(x[:,0],x[:,1],c = y,cmap=cmap_bold)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())

plt.show()
# nbrs = NearestNeighbors(n_neighbors=3,algorithm='kd_tree').fit(x)
# distances,indices = nbrs.kneighbors(x)

