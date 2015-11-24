from nolearn.lasagne import NeuralNet
from nolearn.lasagne import PrintLayerInfo
from lasagne.layers import *
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, label_binarize
from utils import get_otto_dataset, plot_roc_curve, \
                    plot_accuracy, plot_loss, \
                    plot_matrix
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 10
filename= "neural_networks"

X, y, n_classes = get_otto_dataset('../TP5/data/train.csv')
X = X.astype(np.float32)
y = y.astype(np.int32)

X_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.25)

def NeuralNetConstructor(num_features): 
    layers0 = [('input', InputLayer), 
                ('hidden1', DenseLayer), 
                ('dropout1', DropoutLayer), 
                ('hidden2', DenseLayer), 
                ('dropout2', DropoutLayer), 
                #('hidden3', DenseLayer), 
                #('dropout3', DropoutLayer), 
                ('output', DenseLayer)]

    net0 = NeuralNet(layers=layers0,
                    input_shape=(None, num_features),
                    hidden1_num_units=500,
                    dropout1_p=0.5,
                    hidden2_num_units=300,
                    dropout2_p=0.3,
                    #hidden3_num_units=200,
                    #dropout3_p=0.2,
                    output_num_units=n_classes,
                    output_nonlinearity=softmax,
                    update=nesterov_momentum,
                    update_learning_rate=0.04,
                    update_momentum=0.9,
                    verbose=1,
                    regression=False,
                    max_epochs=EPOCHS
                    )
    return net0

net0 = NeuralNetConstructor(93)
net0.fit(X_train, y_train)
predicted = net0.predict(X_test)

"""Testing A Simple Prediction"""
#print("Feature vector: %s" % X_test[:1])
print("Label: %s" % str(y_test[0]))
print("Predicted: %s" % str(net0.predict(X_test[:1])))


"""Metrics"""

# layer_info = PrintLayerInfo()
# net0.verbose = 3
# net0.initialize()
#print layer_info(net0)

print "[Classification Report]: "
print classification_report(y_test, predicted)
print "[Train dataset] Score: ", net0.score(X_train, y_train)
print "[Test dataset] Score: ", net0.score(X_test, y_test)
plot_matrix(net0, X_test, y_test, filename)

valid_accuracies = np.array([i["valid_accuracy"] for i in net0.train_history_])
plot_accuracy(valid_accuracies, filename)

train_loss = [row['train_loss'] for row in net0.train_history_]
valid_loss = [row['valid_loss'] for row in net0.train_history_]
plot_loss(valid_loss, train_loss, filename)

y_score = net0.predict_proba(X_test) #[:, 1]
y_test_bin = np.array(label_binarize(y_test, classes=np.unique(y)))
n_classes = y_test_bin.shape[1]
plot_roc_curve(n_classes, y_test_bin, y_score, filename=filename)