# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,BatchNormalization,UpSampling2D,LeakyReLU, Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import config.srresnet_config as config

class SRResNet:

    def __init__(self):

        # Input dimension
        self.HR_shape = config.HR_SHAPE
        self.LR_shape = config.LR_SHAPE

        # optimizer
        optimizer = Adam(0.0002, 0.5)

        self.SRResNet = self.build()
        self.SRResNet.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    def build(self):
        input = Input(shape=self.LR_shape)

        conv1 = Conv2D(64, (3, 3), strides=1, padding="same", activation='relu')(input)

        def residual_blocks(x):
            # first set of conv
            x1 = Conv2D(64, (3, 3), strides=1, padding="same")(x)
            x1 = BatchNormalization()(x1)
            x1 = Activation("relu")(x1)

            # second set of conv
            x2 = Conv2D(64, (3, 3), strides=1, padding="same")(x1)
            x2 = BatchNormalization()(x2)

            y = add([x, x2])
            return y

        r = residual_blocks(conv1)
        for i in range(15):
            r = residual_blocks(r)

        # next set of conv
        conv2 = Conv2D(64, (3, 3), strides=1, padding='same')(r)
        conv2 = BatchNormalization()(conv2)
        conv2 = add([conv2, conv1])

        # same-size output
        output = Conv2D(3, (1, 1), padding='valid', strides=1)(conv2)

        return Model(input, output)
