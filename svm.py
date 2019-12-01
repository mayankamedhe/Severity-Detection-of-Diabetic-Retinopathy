from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pickle as pkl
import numpy as np

clf = svm.SVC(kernel='linear', class_weight = 'balanced')
with open("features_hog.pk", "rb") as f:
	d = pkl.load(f)

data = d["data"]
data = data / np.linalg.norm(data, axis = 1).reshape(-1,1) #normalizing
labels = d["labels"]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 2)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)
print(y_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

print("Accuracy: ", accuracy)
            