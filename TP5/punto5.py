from sklearn.cross_validation import train_test_split
from sklearn.metrics import adjusted_rand_score, r2_score, mean_squared_error
from sklearn import svm
from pandas import read_csv
from prettytable import PrettyTable
from utils import preprocess_data


#Obtenemos los datos de entrenamiento
train = read_csv('data/1.5/train.csv')
data = train.ix[:, train.columns != 'Hazard'][:1000] #Quitamos la columna Hazard
X = data.ix[:, data.columns != 'Id'][:1000] #Quitamos la columna Id
y = train['Hazard'][:1000]

X_train, X_test, y_train, y_test = train_test_split(X, y)

df_train = preprocess_data(X_train)

svr = svm.SVR(kernel='linear')
svr.fit(df_train.values, y_train)

df_test = preprocess_data(X_test)

predicted_values = svr.predict(df_test.values)

pt = PrettyTable()
pt.add_column("Predicted hazard", predicted_values)
print pt

#Regression score
print "R2 Score"
print r2_score(y_test, predicted_values)
