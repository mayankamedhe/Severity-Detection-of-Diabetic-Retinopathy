from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pickle as pkl

with open("features_hog.pk", "rb") as f:
	d = pkl.load(f)
X = d["data"]## from feature
Y = d["labels"]## from csv file

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

digreg = linear_model.LogisticRegression()
digreg.fit(X_train, y_train)

y_pred = digreg.predict(X_test)

print("Accuracy of Logistic Regression model is:", metrics.accuracy_score(y_test, y_pred)*100)