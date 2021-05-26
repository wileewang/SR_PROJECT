# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,BatchNormalization,UpSampling2D,LeakyReLU, Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import config.srgan_config as config

class SRGAN:

    def __init__(self):

        # Input dimension
        self.HR_shape = config.HR_SHAPE
        self.LR_shape = config.LR_SHAPE

        self.vgg = self.build_vgg()
        self.vgg.trainable = False

        # build two model
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # optimizer
        optimizer = Adam(0.00002, 0.5)

        # define a input
        img_lr = Input(shape=self.LR_shape)
        img_hr = Input(shape=self.HR_shape)

        # extract image features from generated img
        gen_hr = self.generator(img_lr)
        features_gen = self.vgg(gen_hr)

        # discriminate generated image from real image
        prob = self.discriminator(gen_hr)

        # build SRGAN
        self.SRGAN = Model([img_lr, img_hr], [prob, features_gen])

        # compile all models
        self.vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.generator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])# need to train it first
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.SRGAN.compile(loss=['binary_crossentropy', 'mse'],
                           loss_weights=[1e-3, 1],
                           optimizer=optimizer)



    def build_vgg(self):
        '''
        build a pre-trained VGG19 model that outputs image features
        :return: a model
        '''

        # use output of 9th layer to calculate perceptual loss
        vgg = VGG19(weights='imagenet')
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.HR_shape)

        img_features = vgg(img)

        return Model(inputs=img, outputs=img_features)

    def build_generator(self):

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

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding="same")(u)
            u = Activation('relu')(u)
            return u

        r = residual_blocks(conv1)
        for i in range(15):
            r = residual_blocks(r)


        # next set of conv
        conv2 = Conv2D(64, (3, 3),strides=1, padding='same')(r)
        conv2 = BatchNormalization()(conv2)
        conv2 = add([conv2, conv1])

        # same-size output
        output = Conv2D(3, (1, 1), padding='valid',strides=1)(conv2)

        # 4x factor output
        # # pixel shuffle
        # u1 = deconv2d(conv2)
        # u2 = deconv2d(u1)
        #
        # # final set of conv
        # output = Conv2D(3, (9, 9), padding='same', strides=1)(u2)

        return Model(input, output)

    def build_discriminator(self):

        input = Input(shape=self.HR_shape)

        def d_block(input, num_filters, stride):
            d = Conv2D(num_filters, (3, 3), strides=stride)(input)
            d = LeakyReLU(alpha=0.2)(d)
            d = BatchNormalization()(d)
            return d

        conv1 = Conv2D(64, (3, 3), strides=1, padding='same')(input)
        conv1 = LeakyReLU(alpha=0.2)(conv1)

        d1 = d_block(conv1, num_filters=64, stride=2)
        d2 = d_block(d1, num_filters=128, stride=1)
        d3 = d_block(d2, num_filters=128, stride=2)
        d4 = d_block(d3, num_filters=256, stride=1)
        d5 = d_block(d4, num_filters=256, stride=2)
        d6 = d_block(d5, num_filters=512, stride=1)
        d7 = d_block(d6, num_filters=512, stride=2)

        # dense layer
        ds = Dense(1024)(d7)
        ds = LeakyReLU(alpha=0.2)(ds)
        prob = Dense(1, activation='sigmoid')(ds)

        return Model(input, prob)
