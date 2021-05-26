from model.srgan import SRGAN
from model.srresnet import SRResNet
from config import srresnet_config as config
from IO.hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.applications import VGG19
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, add
from tensorflow.keras.models import Model
import numpy as np

# solve failed to get convolution algorithm
import tensorflow as tf
cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=cfg)

def super_res_generator(inputDataGen, targetDataGen):
    # start an infinite loop for the training data
    while True:
        # grab the next input images and target outputs, discarding
        # the class labels (which are irrelevant)
        inputData = next(inputDataGen)[0]
        targetData = next(targetDataGen)[0]

        # yield a tuple of the input data and target data
        yield (inputData, targetData)

inputs = HDF5DatasetGenerator(config.INPUTS_DB, config.BATCH_SIZE)
targets = HDF5DatasetGenerator(config.OUTPUTS_DB, config.BATCH_SIZE)

# print("[INFO] compiling model...")
# vgg = VGG19(weights='imagenet')
# vgg.outputs = [vgg.layers[9].output]
# img = Input(shape=(224, 224, 3))
#
# img_features = vgg(img)
# model = Model(img, img_features)

# train the generator
model = SRResNet().SRResNet
print(model.summary())

#
# not use generator, directly feed all data into GPU

H = model.fit_generator(
    super_res_generator(inputs.generator(), targets.generator()),
    steps_per_epoch=inputs.numImages // config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["loss"],
         label="loss")
plt.title("Loss on super resolution training batch-size:2")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig(config.PLOT_PATH)

# close the HDF5 datasets
inputs.close()
targets.close()

