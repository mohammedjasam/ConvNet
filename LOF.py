import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
# print(__doc__)

np.random.seed(42)

# # Generate train data
# X = 0.3 * np.random.randn(100, 2)
#
# # Generate some abnormal novel observations
# X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
#
# X = np.r_[X + 2, X - 2, X_outliers]
#


# X = np.r_[[3,0.1667],
            #   [4, 0.1667], [5, 0.25],[3, 0.0833], [3, 0.1667],[0,0]]

X = [(3,0.1667),(4, 0.1667), (5, 0.25),(3, 0.0833), (3, 0.1667),(0,0)]
# X=X.reshape(-1,1)
print(X)
# fit the model
clf = LocalOutlierFactor(n_neighbors=1)
y_pred = clf.fit_predict(X)
print(y_pred)
y_pred_outliers = y_pred[:]
print(y_pred_outliers)


# plot the level sets of the decision function
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Local Outlier Factor (LOF)")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

a = plt.scatter(X[:200, 0], X[:200, 1], c='white')
b = plt.scatter(X[200:, 0], X[200:, 1], c='red')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a, b],
           ["normal observations",
            "abnormal observations"],
           loc="upper left")
plt.show()


# """
# =================================================
# Anomaly detection with Local Outlier Factor (LOF)
# =================================================
#
# This example presents the Local Outlier Factor (LOF) estimator. The LOF
# algorithm is an unsupervised outlier detection method which computes the local
# density deviation of a given data point with respect to its neighbors.
# It considers as outlier samples that have a substantially lower density than
# their neighbors.
#
# The number of neighbors considered, (parameter n_neighbors) is typically
# chosen 1) greater than the minimum number of objects a cluster has to contain,
# so that other objects can be local outliers relative to this cluster, and 2)
# smaller than the maximum number of close by objects that can potentially be
# local outliers.
# In practice, such informations are generally not available, and taking
# n_neighbors=20 appears to work well in general.
# """
#
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.neighbors import LocalOutlierFactor
# print(__doc__)
#
# np.random.seed(42)
#
# # # Generate train data
# # X = 0.3 * np.random.randn(100, 2)
# # # Generate some abnormal novel observations
# # X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
# # X = np.r_[X + 2, X - 2, X_outliers]
#
# #--------------generate  data---------------
# # X = np.r_([3,0.1667], [4, 0.1667], [5, 0.25], [3, 0.0833], [3, 0.1667], [0,0], dtype = float)
#
# # X = [[3,0.1667], [4, 0.1667], [5, 0.25], [3, 0.0833], [3, 0.1667], [0,0]]
#
# X = np.array([(3,0.1667),
#               (4, 0.1667), (5, 0.25),(3, 0.0833), (3, 0.1667),(0,0)], dtype=np.float)
#
# # X = np.array([[0,0],
# #               [0,1], [1,1],[3, 0]], dtype=np.float)
#
#
#
# # print (X_outliers)
#
# print(X)
#
#
#
# # fit the model
# clf = LocalOutlierFactor(n_neighbors=1)
# y_pred = clf.fit_predict(X)
# y_pred_outliers = y_pred[10:]
# #print (Xfit)
# print (y_pred)
# print (y_pred_outliers)
#
# # plot the level sets of the decision function
# xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
# Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# plt.title("Local Outlier Factor (LOF)")
# plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
#
# a = plt.scatter(X[:200, 0], X[:200, 1], c='white')
# b = plt.scatter(X[200:, 0], X[200:, 1], c='red')
# plt.axis('tight')
# plt.xlim((-5, 5))
# plt.ylim((-5, 5))
# plt.legend([a, b],
#            ["normal observations",
#             "abnormal observations"],
#            loc="upper left")
# plt.show()
