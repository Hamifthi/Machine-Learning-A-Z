import keras
from keras.models import Sequential
from keras.layers import Dense

def buildClassifier (optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation = "relu", input_dim = 11, units = 6, kernel_initializer = 'uniform'))
    classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = 'uniform'))
    classifier.add(Dense(activation = "sigmoid", units = 1, kernel_initializer = 'uniform'))
    classifier.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return classifier

if __name__ == '__main__':
    print('Working fine')