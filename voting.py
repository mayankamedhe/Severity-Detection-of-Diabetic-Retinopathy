from sklearn.ensemble import  VotingClassifier, RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.svm import SVC
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split


rf = RandomForestClassifier(random_state=1)
svc = SVC(random_state = 1)
mlr = LogisticRegression(random_state = 1)

# rg = RidgeClassifier()
# clf_array = [svc,mlr]


with open("features_hog.pk", "rb") as f:
  d = pkl.load(f)

data = d["data"]
labels = d["labels"]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.4, random_state = 2)

# If ‘hard’, uses predicted class labels for majority rule voting. 
# Else if ‘soft’, predicts the class label based on the argmax of the
#  sums of the predicted probabilities, 
# which is recommended for an ensemble of well-calibrated classifiers.

model = VotingClassifier(estimators=[('rf',rf),( 'svc', svc), ('mlr', mlr)], voting='hard')
model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)
print("test_accuracy",accuracy)