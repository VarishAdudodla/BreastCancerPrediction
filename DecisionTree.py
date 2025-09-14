import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from dataprep import load_data

X_train, X_test, y_train, y_test = load_data()

decmodel = DecisionTreeClassifier()
decmodel.fit(X_train, y_train)
y_pred = decmodel.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

