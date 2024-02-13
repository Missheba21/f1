# step 1
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

#from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten




## step 2
train_dir="f1/.github/workflows/Original_Images/"
generator = ImageDataGenerator()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)  # Optionally, apply transformations here

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=3,
    class_mode='binary'  # or 'categorical' if you have multiple classes
)

# You can then iterate over train_generator to get batches of data



train_ds = generator.flow_from_directory(train_dir,target_size=(224, 224),batch_size=3)
classes = list(train_ds.class_indices.keys())

