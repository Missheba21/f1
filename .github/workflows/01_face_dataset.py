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

from PIL import Image
import os

# Path to the folder containing image files
folder_path = "../f1/.github/workflows/Original_Images/Akshay_Kumar/"



# List all files in the folder
files = os.listdir(folder_path)

# Iterate through each file in the folder
for file_name in files:
    # Check if the file is an image (you can add more image formats if needed)
    if file_name.endswith(".jpg"):
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, file_name)
        
        # Load the image using Pillow
        image = Image.open(image_path)
        
        # Now you can work with the image, for example, display it
        image.show()





