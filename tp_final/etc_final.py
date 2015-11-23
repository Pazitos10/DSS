from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from scipy import interp
from utils import get_otto_dataset, plot_decision_area, \
                  plot_matrix, plot_roc_curve, \
                  plot_learning_curve, \
                  plot_validation_curve
import matplotlib.pyplot as plt
import numpy as np

ESTIMATORS = 10
filename = "extra_trees_classifiers"

X, y, n_classes = get_otto_dataset('../TP5/data/train.csv')

X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, 
                                                    y, 
                                                    test_size=.25)

etc_3 = ExtraTreesClassifier( max_features=0.42, 
                              n_estimators=ESTIMATORS, 
                              n_jobs=-1)

"""Classification Report & Confusion Matrix"""
etc_3.fit(X_train, y_train)
y_pred_etc_3 = etc_3.predict(X_test)

print "[Classification Report]"
print classification_report(y_test, y_pred_etc_3)
print "[Train dataset] Score: %.5f" % etc_3.score(X_train, y_train)
print "[Test dataset] Score: %.5f" % etc_3.score(X_test, y_test)
plot_matrix(etc_3, X_test, y_test, filename)

"""Plot ROC Curve"""
y_score = etc_3.predict_proba(X_test)
y_test_bin = np.array(label_binarize(y_test, classes=np.unique(y)))
n_classes = y_test_bin.shape[1]
plot_roc_curve(n_classes, y_test_bin, y_score, filename)

"""Plot Decision Area"""
clf = ExtraTreesClassifier(n_estimators=ESTIMATORS, max_features=0.42, n_jobs=-1)
plot_decision_area(clf, X_scaled[:, 2:4], y, title="Extra Trees Classifier", filename=filename)

"""Plot Learning Curve"""
X_lc = X_scaled[:10000]
y_lc = y[:10000]
plot_learning_curve(clf, "Extra Trees Classifier", X_lc, y_lc, filename=filename)

"""Plot Validation Curve: max_depth"""
clf = ExtraTreesClassifier(n_estimators=ESTIMATORS ,max_depth=8)
param_name = 'max_depth'
param_range = [1, 2, 4, 8, 16, 32, 100]
plot_validation_curve(clf, X_lc, y_lc,
                  param_name, param_range, 
                  scoring='roc_auc', 
                  cv=n_classes,
                  filename=filename)

"""Plot Validation Curve: n_estimators"""
# clf = ExtraTreesClassifier(n_estimators=ESTIMATORS ,max_features=0.42, max_depth=16)
# param_name = 'n_estimators'
# param_range = [1, 2, 4, 10, 30]
# plot_validation_curve(clf, X_scaled, y,
#                   param_name, param_range, 
#                   scoring='accuracy', cv=n_classes,
#                   filename)