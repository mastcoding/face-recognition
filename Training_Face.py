import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras import models
from keras import optimizers
import cv2
import glob
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import warnings
from keras.callbacks import ModelCheckpoint 
# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
# Suppress general Python warnings (optional)
warnings.filterwarnings('ignore')
def get_files(directory):
  if not os.path.exists(directory):
    return 0
  count=0
  # crawls inside folders
  for current_path,dirs,files in os.walk(directory):
    for dr in dirs:
      count+= len(glob.glob(os.path.join(current_path,dr+"/*")))
  return count
train_data_dir = 'Train_Images'
train_samples =get_files(train_data_dir)
num_classes=len(glob.glob(train_data_dir+"/*")) 
print(num_classes,"Classes")
print(train_samples,"Train images")

test_datagen=ImageDataGenerator(rescale=1./255)
input_shape=(224,224,3)
train_generator =test_datagen.flow_from_directory(train_data_dir,target_size=(224,224),batch_size=32,
                                               class_mode="categorical")
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Convolution2D,Dense,MaxPool2D,Activation,Dropout,Flatten
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
import tensorflow as tf
model = Sequential([
    Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
    BatchNormalization(),
    MaxPool2D(pool_size=(3,3), strides=(2,2)),
    Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPool2D(pool_size=(3,3), strides=(2,2)),
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPool2D(pool_size=(3,3), strides=(2,2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.summary()
categories = list(train_generator.class_indices.keys())
print(categories)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_checkpoint_callback = ModelCheckpoint(
    filepath='Myface_Model.h5',  # Path to save the model
    monitor='accuracy',          # Monitor validation accuracy
    mode='max',                      # Save when validation accuracy is maximized
    save_best_only=True,             # Save only the best model
    verbose=1                        # Display progress
)
from keras.callbacks import ReduceLROnPlateau
history = model.fit(    
    train_generator,
    steps_per_epoch=None,
    epochs=10,
    validation_data=train_generator,
    validation_steps=4,
    verbose=1,
    callbacks=[model_checkpoint_callback,ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3, min_lr=0.00001)],
    shuffle=True)

