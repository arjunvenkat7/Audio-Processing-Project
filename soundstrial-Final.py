# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 02:43:08 2018

@author: Arjun
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Activation, Dropout, BatchNormalization, Flatten
#from keras.layers import Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import SGD, Adam
import keras
from keras.callbacks import EarlyStopping
#from keras.utils import to_categorical
#Using theano backend and using GPU for better performance
import theano.gpuarray
theano.gpuarray.use("cuda0")
# Loading the data
folder = "C:/Users/Arjun/Desktop/cs/"
X_train_org = np.load(folder + 'X_train.npy')
X_test = np.load(folder + 'X_test.npy')
y_labels_train = pd.read_csv(folder + 'y_train.csv', sep=',')['scene_label'].tolist()
# Label lists
labels = sorted(list(set(y_labels_train)))
label2int = {l:i for i, l in enumerate(labels)}
# Map y_train to int labels
y_train_org = keras.utils.to_categorical([label2int[l] for l in y_labels_train])
# Adding contents from cross validation to respective data sets
splitlist = pd.read_csv(folder + 'crossvalidation_train.csv', sep=',')['set'].tolist()
X_train = np.array([x for i, x in enumerate(X_train_org) if splitlist[i] == 'train'])
X_valid = np.array([x for i, x in enumerate(X_train_org) if splitlist[i] == 'test'])
y_train = np.array([y for i, y in enumerate(y_train_org) if splitlist[i] == 'train'])
y_valid = np.array([y for i, y in enumerate(y_train_org) if splitlist[i] == 'test'])
# Normalize dataset
value_max = np.max(np.vstack([X_train, X_valid, X_test]))
X_train = X_train / value_max
X_valid = X_valid / value_max
X_test = X_test / value_max
X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

batch_size = 32#No particular rationale for this 
num_classes = len(labels)#Based on the length of label we obtain num classes

def model_cnn_sound(input_shape):
    model = Sequential()
    model.add(Conv2D(32, 64, input_shape=input_shape,strides=(3,4), activation='relu', padding='same'))#Conv1
    model.add(MaxPooling2D(8, strides=(1,2)))
    model.add(Conv2D(64, 32, strides=(3,3), activation='relu', padding='same'))#conv2
    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, strides=1, activation='relu', padding='same'))#conv3
    model.add(Conv2D(128, 3, strides=1, activation='relu', padding='same'))#conv4
    model.add(Conv2D(256, 3, strides=1, activation='relu', padding='same'))#conv5
    model.add(MaxPooling2D(3, strides=1))
    model.add(Flatten())
    model.add(Dense(num_classes, activation = 'softmax'))#adding dense layer
    #model.add(Dropout(0.5))# We can add a dropout if needed
    return model    

datagen = ImageDataGenerator(
    featurewise_center=True,  
    featurewise_std_normalization=True,  
    rotation_range=0,#Degree shift rotations  
    width_shift_range=0.6,#Range for horizontal shifts  
    height_shift_range=0,#Range for vertical shifts  
    horizontal_flip=True #Random flip horizontally 
)
test_datagen = ImageDataGenerator(
    featurewise_center=True,  
    featurewise_std_normalization=True,  
)
datagen.fit(X_train)
model = model_cnn_sound(X_train.shape[1:])
model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              #optimizer=sgd,
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])
'''history = model.fit(X_train,y_train,
                        validation_data=(X_valid,y_valid),
                        epochs=250,
                        verbose=1)'''#code without data augmentation
'''callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='bestmod.h5', monitor='val_loss', save_best_only=True)]'''
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    epochs=250,
                    #callbacks = callbacks,
                    validation_data=test_datagen.flow(X_valid, y_valid),
                    validation_steps = 100)
print(history.history.keys())
# summarize history for accuracy and plotting between val accuracy and corresponding epoch
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#Obtaining the best accuracy from the model
accuracy = history.history['val_acc']
print(max(accuracy))