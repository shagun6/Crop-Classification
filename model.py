from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
#from tensorflow.keras.applications.resnet152V2 import ResNet152V2
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
#import matplotlib.pyplot as plttf.__version__

import pathlib
dataset_url = "/home/shagunaggarwal/Downloads/kag2"
test_url ="/home/shagunaggarwal/Downloads/crop_images"

batch_size = 32
img_height = 180
img_width = 180

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# re-size all the images to this
IMAGE_SIZE = [180, 180]

import tensorflow
VGG19=tensorflow.keras.applications.VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in VGG19.layers:
    layer.trainable = False

# our layers - you can add more if you want
x = Flatten()(VGG19.output)

prediction = Dense(5, activation='softmax')(x)

# create a model object
model = Model(inputs=VGG19.input, outputs=prediction)

model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory(dataset_url,
                                                 target_size = (180,180),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(test_url,
                                            target_size = (180, 180),
                                            batch_size = 32,
                                            class_mode = 'categorical')
# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

import matplotlib.pyplot as plt

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

from tensorflow.keras.models import load_model

model.save('/home/shagunaggarwal/Desktop/crop_predict/models/model_vgg19.h5')
y_pred = model.predict(test_set)
import numpy as np
y_pred = np.argmax(y_pred, axis=1)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model('/home/shagunaggarwal/Desktop/crop_predict/models/model_vgg19.h5')

# !pip install flask_ngrok


