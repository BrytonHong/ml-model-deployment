import numpy as np
import pandas as pd


training_data = pd.read_csv('storepurchasedata.csv')

training_data.describe()
 
X = training_data.iloc[:,:-1].values

Y = training_data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

classifier.fit(X_train, Y_train)


Y_pred = classifier.predict(X_test)
Y_prob = classifier.predict_proba(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))


from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

new_prediction = classifier.predict(sc.transform(np.array([[40,20000]])))

new_prediction_proba = classifier.predict_proba(sc.transform(np.array([[40,20000]])))[:,1]

new_pred = classifier.predict(sc.transform(np.array([[42,50000]])))

new_pred_proba = classifier.predict_proba(sc.transform(np.array([[42,50000]])))[:,1]


import pickle

model_file ='classifier.pickle'
pickle.dump(classifier, open(model_file,'wb'))

scaler_file = "sc.pickle"

pickle.dump(sc, open(scaler_file,'wb'))

