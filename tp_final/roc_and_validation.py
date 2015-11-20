import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from scipy import interp
import pandas as pd
from sklearn.decomposition import RandomizedPCA as PCA

N_ESTIMATORS = 900  #100
N_COMPONENTS = 57 #57

df = pd.read_csv('../TP5/data/train.csv')
X = df.ix[:,1:-1].values.astype(np.float32)
y = df['target']

# Binarize the output
y = label_binarize(y, classes=range(9))
n_classes = y.shape[1]
print n_classes

X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled.astype(np.float32), 
                                                     y, test_size=.5)

et = ExtraTreesClassifier(n_estimators=200, max_features=0.42, max_depth=None, min_samples_split=10, random_state=0)

et.fit(X_train, y_train)
print "score train dataset"
print et.score(X_train, y_train)

print "score test dataset"
print et.score(X_test, y_test)


etc_3 = ExtraTreesClassifier(max_features=0.42, n_estimators=N_ESTIMATORS, n_jobs=2, oob_score=True, bootstrap=True)
etc_3.fit(X_train, y_train)
y_score = etc_3.oob_decision_function_

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[i][:, 0])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# ##############################################################################
# # Plot of a ROC curve for a specific class
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


# ##############################################################################
# # Plot ROC curves for the multiclass problem

# # Compute macro-average ROC curve and ROC area

# # First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
#plt.figure()
#plt.plot(fpr["micro"], tpr["micro"],
#         label='micro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["micro"]),
#         linewidth=2)

#plt.plot(fpr["macro"], tpr["macro"],
#         label='macro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["macro"]),
#         linewidth=2)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


from sklearn.learning_curve import validation_curve


def plot_validation_curve(estimator, X, y, param_name, param_range,
                      ylim=(0, 1.1), cv=5, n_jobs=2, scoring=None):
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
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.semilogx(param_range, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.semilogx(param_range, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    print("Best test score: {:.4f}".format(test_scores_mean[-1]))

    plt.show()


clf = ExtraTreesClassifier(max_features=0.42, n_jobs=2)
param_name = 'n_estimators'
param_range = [10, 20, 40, 80, 100]
plot_validation_curve(clf, X_train, y_train, param_name, param_range)