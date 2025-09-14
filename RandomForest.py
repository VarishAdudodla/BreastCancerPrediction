import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from dataprep import load_data

X_train, X_test, y_train, y_test = load_data()

rfmodel = RandomForestClassifier(n_estimators=100)
rfmodel.fit(X_train, y_train)
y_pred = rfmodel.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))