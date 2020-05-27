import keras
from keras import optimizers
from keras.models import Sequential 
from keras.layers import Dropout, Flatten, Dense 
def create_tf_model(num_classes):
    model = Sequential() 
    model.add(Flatten(input_shape=(7, 7, 512))) 
    model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3))) 
    model.add(Dropout(0.5)) 
    model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3))) 
    model.add(Dropout(0.3)) 
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
    return model