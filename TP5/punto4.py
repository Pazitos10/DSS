import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import svm, grid_search
import numpy as np

white_wine_data = pd.read_csv('data/1.4/winequality-white.csv')

X = white_wine_data[1:-1].values.astype(np.float32)[:1000]
y = white_wine_data['quality'].values.astype(np.float32)[:1000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

clasificador = svm.SVC(kernel='rbf', C=1, gamma=0.001, cache_size=500)
clasificador.fit(X_train, y_train)

predicted = clasificador.predict(X_test)
from sklearn.metrics import classification_report

print classification_report(y_test, predicted)