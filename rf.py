from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pickle as pkl
import numpy as np
from sklearn.ensemble import RandomForestClassifier

with open("features_hog.pk", "rb") as f:
	d = pkl.load(f)
data = d["data"]
data = data / np.linalg.norm(data, axis = 1).reshape(-1,1)
labels = d["labels"]


clf = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.4, random_state = 2)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)