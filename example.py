import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers,datasets,layers
import numpy as np
import os
from matplotlib import pyplot as plt


#data preparation
(train_x,train_y),(test_x,test_y)=datasets.mnist.load_data()

num_classes=10
n_train=train_x.shape[0]
n_test=test_x.shape[0]
n_features=784




train_y=keras.utils.to_categorical(train_y,num_classes)
test_y=keras.utils.to_categorical(test_y,num_classes)

train_x=train_x.reshape(n_train,28,28,1)
test_x=test_x.reshape(n_test,28,28,1)

train_x=train_x.astype('float32')/255
test_x=test_x.astype('float32')/255


#2conv and 2 pooling layers

model=keras.Sequential()
model.add(layers.Conv2D(filters=32,kernel_size=5,strides=(1,1),padding='same'
                             ,activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='valid'))

model.add(layers.Conv2D(filters=64,kernel_size=3,strides=(1,1),padding='same'
                             ,activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='valid'))
#for overfitting
model.add(layers.Dropout(0.25))
#to make it a vector
model.add(layers.Flatten())

#create the hidden laters (normal preceptrone)
model.add(layers.Dense(128,activation=tf.nn.relu))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizers='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=train_x,y=train_y,epochs=5,batch_size=128,verbose=2)

score=model.evaluate(test_x,test_y)

model.save('final_CNN_model.h5')

model.summary()

from tensorflow.keras.models import load_model
new_model=load_model('final_CNN_model.h5')
new_model.summary()


import matplotlib.pyplot as plt
def visual(n):
    final=new_model.predict_classes(test_x[0:n])
    figure,ax=plt.subplots(nrows=int(n/5),ncols=5)
    ax=ax.flatten()
    print('prediction results of the first {} images :'.format(n))
    for i in range(n):
        print(final[i],end=',')
        if int((i+1)%5)==0:
            print('\t')
        img=test_x[i].reshape((28,28))
        plt.axis("off")
        ax[i].imshow(img,cmap='Greys',interpolation='nearest')
        ax[i].axis("off")
    print('first {} images in the test set : '.format(n))

visual(20)
