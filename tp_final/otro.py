from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.visualize import plot_loss
from nolearn.metrics import  multiclass_logloss, LearningCurve
from lasagne.layers import *
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax
from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from nolearn.lasagne import PrintLayerInfo
import numpy as np
import theano

MAX_EPOCHS_SPACE = 110 # total space the decreasing learning rates and momentum will be stretched along
MAX_EPOCHS = 100       # actual rounds of running 
BATCH_SIZE = 500

df = read_csv('../TP5/data/train.csv')
X = df.ix[:,1:-1].values.astype(np.float32)
y = df['target']

le = LabelEncoder()
y_encoded = le.fit_transform(y).astype(np.int32)

X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)

#y_scaler = MinMaxScaler()
#y_scaled = y_scaler.fit_transform(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded)

def NeuralNetConstructor(num_features): 
    layers0 = [('input', InputLayer), 
                ('hidden1', DenseLayer), 
                ('dropout1', DropoutLayer), 
                ('hidden2', DenseLayer), 
                #('dropout2', DropoutLayer), 
                #('hidden3', DenseLayer), 
                #('dropout3', DropoutLayer), 
                ('output', DenseLayer)]

    net0 = NeuralNet(layers=layers0,
                    input_shape=(None, num_features),
                    hidden1_num_units=500,
                    dropout1_p=0.3,
                    hidden2_num_units=250,
                    #dropout2_p=0.3,
                    #hidden3_num_units=200,
                    #dropout3_p=0.2,
                    output_num_units=le.classes_.shape[0],
                    output_nonlinearity=softmax,
                    update=nesterov_momentum,
                    update_learning_rate=theano.shared(np.float32(0.005)), #0.003
                    update_momentum=theano.shared(np.float32(0.9)),
                    verbose=1,
                    regression=False,
                    batch_iterator_train = BatchIterator(batch_size = BATCH_SIZE),
                    max_epochs=MAX_EPOCHS
                    # on_epoch_finished=[
                    #        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                    #        AdjustVariable('update_momentum', start=0.9, stop=0.999)]
                    )
    return net0

net0 = NeuralNetConstructor(93)
net0.fit(X_train, y_train)

layer_info = PrintLayerInfo()
net0.verbose = 3
net0.initialize()
layer_info(net0)
print layer_info

predicted = net0.predict(X_test)

plot_loss(net0)
plt.savefig('loss.png')

print classification_report(y_test, predicted)
valid_accuracies = np.array([i["valid_accuracy"] for i in net0.train_history_])
plt.clf()
plt.plot(valid_accuracies, label="accuracy")
plt.ylim([0,1])
plt.savefig('accuracy.png')