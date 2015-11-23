import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.learning_curve import learning_curve, validation_curve
from nolearn.lasagne.visualize import plot_loss
from scipy import interp
import numpy as np
from pandas import read_csv


def get_otto_dataset(dataset_path):
    df = read_csv(dataset_path)
    X = df.ix[:,1:-1]
    y = df['target']

    le = LabelEncoder()
    y = le.fit_transform(y)
    n_classes = len(le.classes_)
    return X, y, n_classes

def save_plot(filename):
    plt.savefig("plots/"+filename)
    print "Saved: ", filename

def plot_matrix(clf, X_test, y_test, filename):
    """Plot Confussion Matrix from a given classifier"""
    plt.clf()
    plt.imshow(confusion_matrix(y_test, clf.predict(X_test)),
               interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    save_plot('confusion_matrix_'+str(filename)+'.png')
    #plt.show()

def plot_decision_area(clf, X, y, title=None, filename=None):
    RANDOM_SEED = 13  # fix the seed on each iteration
    n_classes = 9
    plot_colors = "ryb"
    cmap = plt.cm.RdYlBu
    plot_step = 0.02  # fine step width for decision surface contours
    plot_step_coarser = 0.5  # step widths for coarse classifier guesses

    clf.fit(X, y)
    score = clf.score(X, y)
    if title:
        model_details = title
    else:
        model_details = "Algorithm"
    plt.title(model_details+" Performance")
    model_details += " with {} estimators".format(len(clf.estimators_))
    print model_details + " and with 2 features has a score of: ", score

    # Now plot the decision boundary using a fine mesh as input to a
    # filled contour plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    estimator_alpha = 1.0 / len(clf.estimators_)
    for tree in clf.estimators_:
        Z = tree.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

    # Build a coarser grid to plot a set of ensemble classifications
    # to show how these are different to what we see in the decision
    # surfaces. These points are regularly space and do not have a black outline
    xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser),
                                         np.arange(y_min, y_max, plot_step_coarser))
    Z_points_coarser = clf.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
    cs_points = plt.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap, edgecolors="none")

    # Plot the training points, these are clustered together and have a
    # black outline
    for i, c in zip(xrange(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=c, cmap=cmap)

    plt.axis("tight")
    save_plot("decision_area_"+str(filename)+".png")
    #plt.show()

def plot_roc_curve(n_classes, y_test, y_score, filename=None):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        #fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score.ravel())
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
     
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    label = 2
    plt.plot(fpr[label], tpr[label], label='ROC curve of class %d (area = %0.2f)' % (label, roc_auc[label]))
    plt.fill_between(fpr[label], tpr[label], alpha='0.2')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    save_plot("roc_curve_label_2_"+str(filename)+".png")
    #plt.show()

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             linewidth=2)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], 
                label='ROC curve of class {0} (area = {1:0.5f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    #plt.fill_between(fpr["micro"], tpr["micro"], alpha='0.2')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")

    save_plot("roc_curves_"+str(filename)+".png")
    #plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=(0, 1.1), cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), 
                        filename=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, 
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    save_plot("learning_curve_"+str(filename)+".png")
    #plt.show()

def plot_validation_curve(estimator, X, y, param_name, param_range,
                      ylim=(0, 1.1), cv=None, n_jobs=-1, scoring=None, 
                      filename=None):
    estimator_name = type(estimator).__name__
    plt.title("Validation curves for %s on %s"
          % (param_name, estimator_name))
    plt.grid()
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.xlim(min(param_range), max(param_range))
    plt.ylim(*ylim) 

    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name, param_range,
        cv=cv, n_jobs=n_jobs, scoring=scoring)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.semilogx(param_range, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.semilogx(param_range, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    print("Best test score: {:.4f}".format(test_scores_mean[-1]))
    save_plot("validation_curve_"+str(filename)+".png")
    #plt.show()

def plot_accuracy(valid_accuracies, filename=None):
    valid_accuracies = valid_accuracies
    plt.clf()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(valid_accuracies, label="accuracy")
    plt.ylim([0,1])
    save_plot('accuracy_'+str(filename)+'.png')

def plot_loss(valid_loss, train_loss, filename=None):
    plt.clf()
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    save_plot('loss_'+str(filename)+'.png')