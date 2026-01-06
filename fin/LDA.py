

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])


clf = LinearDiscriminantAnalysis(
  n_components=1, 
  priors=None, 
  shrinkage=None, 
  solver='svd')

print(X)
print(y)
clf.fit(X, y)
print(clf.predict([[-0.8, -1]]))


# n_components:需要降维時降到的维数，1~類別數-1 之間。
# priors:用來指定總體上每組出現的機率，即先驗機率。
# shrinkage:收縮率 None,'auto',float *當solver為'lsqr' or 'eigen'才可使用'auto'or float
# solver:求解器 'svd'奇異值分解（singular value decomposition）,'lsqr'最小二乘法（least squares method）,'eigen'特徵分解（Eigendecomposition）

