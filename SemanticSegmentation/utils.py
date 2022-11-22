""" Aux functions for unet semantic segmentation

INDEX:
->DataGeneration class
->build_unet function

"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras as keras
from tensorflow.keras import backend as K


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(256,256), n_channels=6,n_classes=1, shuffle=True,forest_index=5,prefix='data_forest_50'):
        'Initialization'
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.forest_index = forest_index
        self.indexes = np.arange(len(self.list_IDs))
        self.prefix = prefix
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_IDs = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_IDs)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_IDs):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        # Initialization
        # original images are shape (channels, x, y)
        # keras needs them to be (batch, x, y, channels)
        # we start we empty arrays in the target shape
        X = np.empty(shape=(self.batch_size, *self.dim, self.n_channels))
        y = np.empty(shape=(self.batch_size, *self.dim, 1)) 
        # Generate data, using a list of IDs
        for i, ID in enumerate(batch_IDs):
            # Load label and images of the same id
            y_ = np.load(self.prefix+'/labels/labels_' + ID)
            X_ = np.load(self.prefix+'/images/image_' + ID)
            X[i,] = np.moveaxis(X_,0,-1) # change axis order (see above)
            # get only the forest (1) vs everything else (0), and change axis order
            y[i,] = (y_==5).astype(np.int).reshape((*self.dim,1))

        return X, y
    
# define a basic convolution block     
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

# define the encoder block
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

# define a decoder block
def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

# put together the blocks to build a unet
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    b1 = conv_block(p4, 512)

    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)
    
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="U-Net")
    return model