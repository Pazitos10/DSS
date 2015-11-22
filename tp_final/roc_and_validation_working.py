import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from scipy import interp
import pandas as pd
import numpy as np
from utils import plot_decision_area, plot_matrix, \
                    plot_validation_curve, plot_roc_curve, \
                    another_plot

N_ESTIMATORS = 100

df = pd.read_csv('../TP5/data/train.csv')
X = df.ix[:,1:-1]
y = df['target']

le = LabelEncoder()
y = le.fit_transform(y)
n_classes = len(le.classes_)

X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.25)

etc_3 = ExtraTreesClassifier(   max_features=0.42,
                                n_estimators=N_ESTIMATORS, 
                                n_jobs=-1, 
                                oob_score=True, 
                                bootstrap=True)

#y_score = etc_3.fit(X_train, y_train).predict_proba(X_test)[:, 1]
#y_pred_etc_3 = etc_3.predict(X_test)

#print "score train dataset"
#print etc_3.score(X_train, y_train)

#print "score test dataset"
#print etc_3.score(X_test, y_test)


#print classification_report(y_test, y_pred_etc_3)
#plot_matrix(etc_3, X_test, y_test)

#y_test_bin = np.array(label_binarize(y_test, classes=np.unique(y)))
#n_classes = y_test_bin.shape[1]

#plot_roc_curve(n_classes, y_test_bin, y_score)

clf = ExtraTreesClassifier(n_estimators=30, max_features=0.42, n_jobs=-1)

plot_decision_area(clf, X_scaled[:, 2:4], y)

#Consume mucha RAM y procesador
# clf = ExtraTreesClassifier(max_features=0.42, n_jobs=-1)
# param_name = 'n_estimators'
# param_range = [30, 40, 100, 200]
# plot_validation_curve(  clf, X_train, y_train, 
#                         param_name, param_range, 
#                         scoring="accuracy", cv=9)



"""
Small Log

21/11/2015 21:51
score train dataset
1.0
score test dataset
0.809243697479
             precision    recall  f1-score   support

          0       0.81      0.43      0.56       482
          1       0.72      0.87      0.79      4087
          2       0.62      0.49      0.54      1992
          3       0.82      0.47      0.60       651
          4       0.98      0.97      0.98       676
          5       0.93      0.95      0.94      3468
          6       0.80      0.60      0.68       735
          7       0.87      0.93      0.90      2152
          8       0.86      0.88      0.87      1227

avg / total       0.81      0.81      0.80     15470

Ver archivos: confusion_etc.png, roc_etc_class_2.png y roc_all_etc.png

======================================================================
21/11/2015 - 22:17 (solo validation_curve)

score train dataset
1.0
score test dataset
0.813510019392
             precision    recall  f1-score   support

          0       0.78      0.43      0.55       474
          1       0.72      0.88      0.79      4069
          2       0.64      0.50      0.56      1990
          3       0.85      0.42      0.56       656
          4       0.97      0.97      0.97       722
          5       0.94      0.95      0.94      3548
          6       0.81      0.60      0.69       693
          7       0.88      0.94      0.91      2074
          8       0.85      0.89      0.87      1244

avg / total       0.81      0.81      0.80     15470

(15470,)
(15470, 9)
0.75
Best test score: 0.8146
[Finished in 849.7s]

"""