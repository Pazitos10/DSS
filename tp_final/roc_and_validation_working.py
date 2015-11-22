from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from scipy import interp
import pandas as pd
import numpy as np
from sklearn.decomposition import RandomizedPCA as PCA
import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve

N_ESTIMATORS = 100  #100
N_COMPONENTS = 57 #57

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
y_score = etc_3.fit(X_train, y_train).predict_proba(X_test)[:, 1]
y_pred_etc_3 = etc_3.predict(X_test)

print "score train dataset"
print etc_3.score(X_train, y_train)

print "score test dataset"
print etc_3.score(X_test, y_test)

from utils import plot_decision_area, plot_matrix
print classification_report(y_test, y_pred_etc_3)
plot_matrix(etc_3, X_test, y_test)



# Compute ROC curve and ROC area for each class
y_test_bin = np.array(label_binarize(y_test, classes=np.unique(y)))
n_classes = y_test_bin.shape[1]

#y_score_bin = label_binarize(y_score, classes=np.unique(y))


print np.array(y_score).shape
print y_test_bin.shape

print y_score[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score.ravel())
    roc_auc[i] = auc(fpr[i], tpr[i])


##############################################################################
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.5f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

def plot_validation_curve(estimator, X, y, param_name, param_range,
                      ylim=(0, 1.1), cv=None, n_jobs=-1, scoring=None):
    estimator_name = type(estimator).__name__
    plt.clf()
    plt.title("Validation curves for %s on %s"
          % (param_name, estimator_name))
    plt.ylim(*ylim); plt.grid()
    plt.xlim(min(param_range), max(param_range))
    plt.xlabel(param_name)
    plt.ylabel("Score")

    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name, param_range,
        cv=cv, n_jobs=n_jobs, scoring=scoring)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.semilogx(param_range, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.semilogx(param_range, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    print("Best test score: {:.4f}".format(test_scores_mean[-1]))

    plt.show()


clf = ExtraTreesClassifier(max_features=0.42, n_jobs=-1)
param_name = 'n_estimators'
param_range = [30, 40, 100, 200]
plot_validation_curve(  clf, X_train, y_train, 
                        param_name, param_range, 
                        scoring="accuracy", cv=9)



"""
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
Ultima corrida - 22:17 - 21/11/2015 (solo validation_curve)

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