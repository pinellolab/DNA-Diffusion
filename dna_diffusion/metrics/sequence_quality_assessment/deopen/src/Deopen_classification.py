'''
This script is used for running Deopen classification model.
Usage:
    THEANO_FLAGS='device=gpu,floatX=float32' python Deopen_classification.py -in <inputfile> -out <outputfile>
    
    inputfile.hkl -- preprocessed file containing different features (hkl format)
    outputfile -- trained model to be saved (hkl format)
'''
import hickle as hkl
import argparse
from sklearn.cross_validation import ShuffleSplit
import numpy as np
import hickle as hkl
import theano
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import SliceLayer
from lasagne.layers import FlattenLayer
from lasagne.layers import ConcatLayer
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
    print 'Using Lasagne.layers.dnn (faster)'
except ImportError:
    from lasagne.layers import Conv2DLayer
    from lasagne.layers import MaxPool2DLayer
    print 'Using Lasagne.layers (slower)'
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import BatchIterator
from sklearn import metrics
floatX = theano.config.floatX

def float32(k):
    return np.cast['float32'](k)

#updating learning rate after each epoch
class AdjustVariable(object):
    def __init__(self, name, start=0.001, stop=0.0001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

#stop training if valid loss doesn't decrease in consecutive epoches         
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

#save the best parameters for each trial           
class SaveTrainHistory(object):
    def __init__(self, iteration):
        self.iteration = iteration
    def __call__(self, nn, train_history):
        val_acc[self.iteration] = train_history[-1]['valid_accuracy']
        params.append(nn.get_all_params_values())
        
#load the best parameters before training        
class LoadBestParam(object):
    def __init__(self, iteration):
        self.iteration = iteration
    def __call__(self, nn, train_history):
        nn.load_params_from(params[self.iteration])

#split the data into training set, testing set
def data_split(inputfile):
    data = hkl.load(inputfile)
    X = data['mat']
    X_kspec = data['kmer']
    y = data['y']
    rs = ShuffleSplit(len(y), n_iter=1,random_state = 1)
    X_kspec = X_kspec.reshape((X_kspec.shape[0],1024,4))
    X = np.concatenate((X,X_kspec), axis = 1)
    X = X[:,np.newaxis]
    X = X.transpose((0,1,3,2))
    for train_idx, test_idx in rs:
        X_train = X[train_idx,:]
        y_train = y[train_idx]
        X_test = X[test_idx,:]
        y_test = y[test_idx]
    X_train = X_train.astype('float32')
    y_train = y_train.astype('int32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('int32')
    return [X_train, y_train, X_test, y_test]

#define the network architecture
def create_network():
    l = 1000
    pool_size = 5
    test_size1 = 13
    test_size2 = 7
    test_size3 = 5
    kernel1 = 128
    kernel2 = 128
    kernel3 = 128
    layer1 = InputLayer(shape=(None, 1, 4, l+1024))
    layer2_1 = SliceLayer(layer1, indices=slice(0, l), axis = -1)
    layer2_2 = SliceLayer(layer1, indices=slice(l, None), axis = -1)
    layer2_3 = SliceLayer(layer2_2, indices = slice(0,4), axis = -2)
    layer2_f = FlattenLayer(layer2_3)
    layer3 = Conv2DLayer(layer2_1,num_filters = kernel1, filter_size = (4,test_size1))
    layer4 = Conv2DLayer(layer3,num_filters = kernel1, filter_size = (1,test_size1))
    layer5 = Conv2DLayer(layer4,num_filters = kernel1, filter_size = (1,test_size1))
    layer6 = MaxPool2DLayer(layer5, pool_size = (1,pool_size))
    layer7 = Conv2DLayer(layer6,num_filters = kernel2, filter_size = (1,test_size2))
    layer8 = Conv2DLayer(layer7,num_filters = kernel2, filter_size = (1,test_size2))
    layer9 = Conv2DLayer(layer8,num_filters = kernel2, filter_size = (1,test_size2))
    layer10 = MaxPool2DLayer(layer9, pool_size = (1,pool_size))
    layer11 = Conv2DLayer(layer10,num_filters = kernel3, filter_size = (1,test_size3))
    layer12 = Conv2DLayer(layer11,num_filters = kernel3, filter_size = (1,test_size3))
    layer13 = Conv2DLayer(layer12,num_filters = kernel3, filter_size = (1,test_size3))
    layer14 = MaxPool2DLayer(layer13, pool_size = (1,pool_size))
    layer14_d = DenseLayer(layer14, num_units= 256)
    layer3_2 = DenseLayer(layer2_f, num_units = 128)
    layer15 = ConcatLayer([layer14_d,layer3_2])
    layer16 = DropoutLayer(layer15,p=0.5)
    layer17 = DenseLayer(layer16, num_units=256)
    network = DenseLayer(layer17, num_units= 2, nonlinearity=softmax)
    return network


#random search to initialize the weights
def model_initial(X_train,y_train,max_iter = 5):
    global params, val_acc
    params = []
    val_acc = np.zeros(max_iter)
    lr = theano.shared(np.float32(1e-4))
    for iteration in range(max_iter):
        print 'Initializing weights (%d/5) ...'%(iteration+1)
        network_init = create_network()
        net_init = NeuralNet(
                network_init,
                max_epochs=3,
                update=adam,
                update_learning_rate=lr,
                train_split=TrainSplit(eval_size=0.1),
                batch_iterator_train=BatchIterator(batch_size=32),
                batch_iterator_test=BatchIterator(batch_size=64),
                on_training_finished=[SaveTrainHistory(iteration = iteration)],
                verbose=0)
        net_init.initialize()
        net_init.fit(X_train, y_train)
        
#model training         
def model_train(X_train, y_train,learning_rate = 1e-4,epochs = 50):
    network = create_network()
    lr = theano.shared(np.float32(learning_rate))
    net = NeuralNet(
                network,
                max_epochs=epochs,
                update=adam,
                update_learning_rate=lr,
                train_split=TrainSplit(eval_size=0.1),
                batch_iterator_train=BatchIterator(batch_size=32),
                batch_iterator_test=BatchIterator(batch_size=64),
                #on_training_started=[LoadBestParam(iteration=val_acc.argmax())],
                on_epoch_finished=[EarlyStopping(patience=5)],
                verbose=1)
    print 'Loading pre-training weights...'
    net.load_params_from(params[val_acc.argmax()])
    print 'Continue to train...'
    net.fit(X_train, y_train)
    print 'Model training finished.'
    return net


#model testing
def model_test(net, X_test, y_test, outputfile):
    #net.load_params_from('saved_weights_file')
    y_pred = net.predict(X_test)
    y_prob = net.predict_proba(X_test)
    print 'Accuracy score is {}'.format(metrics.accuracy_score(y_test, y_pred))
    print 'ROC AUC score is {}'.format(metrics.roc_auc_score(y_test, y_prob[:,-1]))
    hkl.dump([y_prob[:,-1],y_test],outputfile)

#save model parameters
def save_model(model,outputfile):
    net.save_params_to(open(outputfile,'w'))


if  __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Deopen classication model') 
    parser.add_argument('-in', dest='input', type=str, help='inputfile')
    parser.add_argument('-out', dest='output', type=str, help='outputfile')
    args = parser.parse_args()
    X_train, y_train, X_test, y_test = data_split(args.input)
    model_initial(X_train,y_train,5)
    model = model_train(X_train, y_train)
    #save_model(model, args.output)
    model_test(model, X_test, y_test,args.output)


