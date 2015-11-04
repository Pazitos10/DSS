from nolearn.lasagne import NeuralNet
from lasagne.layers import *
from lasagne.update_momentum import nesterov_momentum
from pandas import read_csv
from sklearn.metrics import r2_score
from utils import preprocess_data

def NeuralNetConstructor(num_features): 
    layers0 = [('input', InputLayer), 
                ('hidden1', DenseLayer), 
                ('dropout1', DropoutLayer), 
                ('hidden2', DenseLayer), 
                ('dropout2', DropoutLayer), 
                ('hidden3', DenseLayer), 
                ('dropout3', DropoutLayer), 
                ('output', DenseLayer)]

    net0 = NeuralNet(layers=layers0,
                    input_shape=(None, num_features),
                    hidden1_num_units=50,
                    dropout1_p=0.3,
                    hidden2_num_units=150,
                    dropout2_p=0.5,
                    hidden3_num_units=200,
                    dropout3_p=0.2,
                    output_num_units=1,
                    output_nonlinearity=None,
                    update=nesterov_momentum,
                    update_learning_rate=0.05,
                    update_momentum=0.9,
                    eval_size=0.1,
                    verbose=1,
                    regression=True,
                    max_epochs=35)
    return net0

train = read_csv('data/1.5/train.csv')
data = train.ix[:, train.columns != 'Hazard'][:1000] #Quitamos la columna Hazard
X = data.ix[:, data.columns != 'Id'][:1000] #Quitamos la columna Id
y = train['Hazard'][:1000]

new_X = preprocess_data(X)

X_train, X_test, y_train, y_test = train_test_split(new_X, y)

net0 = NeuralNetConstructor(32)
net0.train(X_train, y_train)

predicted = net0.predict(X_test)

print r2_score(y_test, predicted)

# R2 > 0