

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=10, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X, y)
print(X)
print(y)
print(clf.predict([[0, 0, 0, 0]]))


#max_depth:深度, 
#random_state表示用於指定隨機數生成器的種子隨機training與testing的模式，可固定模式
#n_estimators: 森林中樹木的數量
#n_informative: 分裂的信息特徵
#n_redundant: 冗餘特徵數量
#max_features: 劃分時考慮的最大特徵數
#shuffle: 打亂隨機排序