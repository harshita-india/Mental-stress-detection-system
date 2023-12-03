import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import datasets
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Loading csv file
data = np.loadtxt(open("features1.csv", "rb"), delimiter=",")
#print(data)
X_data = data[:,0:31]
y_data = data[:,32]
#print(X_data)
#print(y_data)

# ploting the graph
plt.figure(figsize=(10, 6))
plt.plot(X_data, y_data, marker='o', linestyle='-')
plt.xlabel('X_data')
plt.ylabel('y_data')
plt.title('Line Plot of X Data vs. Y Data')
plt.grid(True)
plt.show()
 
 
# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, random_state=1)

# DECISION TREE CLASSIFIER
dt = DecisionTreeClassifier(random_state=0)
# train the model
dt.fit(X_train, y_train)
# make predictions
dt_pred = dt.predict(X_test)
# print the accuracy
print("Accuracy of Decision Tree Classifier: ",
      accuracy_score(y_test, dt_pred))
# print other performance metrics
print("Precision of Decision Tree Classifier: ",
      precision_score(y_test, dt_pred, average='weighted'))
print("Recall of Decision Tree Classifier: ",
      recall_score(y_test, dt_pred, average='weighted'))
print("F1-Score of Decision Tree Classifier: ",
      f1_score(y_test, dt_pred, average='weighted'))
 
# GAUSSIAN NAIVE BAYES
gnb = GaussianNB()
# train the model
gnb.fit(X_train, y_train)
# make predictions
gnb_pred = gnb.predict(X_test)
# print the accuracy
print("Accuracy of Gaussian Naive Bayes: ",
      accuracy_score(y_test, gnb_pred))
# print other performance metrics
print("Precision of Gaussian Naive Bayes: ",
      precision_score(y_test, gnb_pred, average='weighted'))
print("Recall of Gaussian Naive Bayes: ",
      recall_score(y_test, gnb_pred, average='weighted'))
print("F1-Score of Gaussian Naive Bayes: ",
      f1_score(y_test, gnb_pred, average='weighted'))

# SUPPORT VECTOR MACHINE
svm_clf = svm.SVC(kernel='linear')  # Linear Kernel
# train the model
svm_clf.fit(X_train, y_train)
# make predictions
svm_clf_pred = svm_clf.predict(X_test)
# print the accuracy
print("Accuracy of Support Vector Machine: ",
      accuracy_score(y_test, svm_clf_pred))
# print other performance metrics
print("Precision of Support Vector Machine: ",
      precision_score(y_test, svm_clf_pred, average='weighted'))
print("Recall of Support Vector Machine: ",
      recall_score(y_test, svm_clf_pred, average='weighted'))
print("F1-Score of Support Vector Machine: ",
      f1_score(y_test, svm_clf_pred, average='weighted'))