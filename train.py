import cv2
import random
from model.srgan import SRGAN
from model.srresnet import SRResNet
from config import srresnet_config as config
from IO.hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.applications import VGG19
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, add
from tensorflow.keras.models import Model
import numpy as np
from imutils import paths

# solve failed to get convolution algorithm
import tensorflow as tf
cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=cfg)

# set random seed
randum = random.randint(0, 100)
random.seed(randum)

# train the generator
model = SRResNet().SRResNet

imageLRPaths = list(paths.list_images(config.IMAGES))
imageHRPaths = list(paths.list_images(config.LABELS))

# sort list to keep sequence the same
imageHRPaths.sort()
imageLRPaths.sort()

print(imageLRPaths)
print(imageHRPaths)

pairPaths = [(x, y) for x, y in zip(imageLRPaths, imageHRPaths)]
random.shuffle(pairPaths)
numOfData = len(imageHRPaths)
print(pairPaths[:int(config.RATIO_DATASET*numOfData)])


inputs = []
targets = []

for (imageLRPath, imageHRPath) in pairPaths:
    input_image = cv2.imread(imageLRPath)
    target_image = cv2.imread(imageHRPath)
    # cv2.imshow('lr', input_image)
    # cv2.imshow('hr', target_image)
    # cv2.waitKey(0)

    inputs.append(input_image)
    targets.append(target_image)

inputs = np.array(inputs)
targets = np.array(targets)



# not use generator, directly feed all data into GPU

H = model.fit(
    x=inputs, y=targets,
    batch_size=2,
    epochs=config.NUM_EPOCHS
)

#
# # save the model to file
# print("[INFO] serializing model...")
# model.save(config.MODEL_PATH, overwrite=True)
#
# # plot the training loss
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["loss"],
#          label="loss")
# plt.title("Loss on super resolution training batch-size:2")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig(config.PLOT_PATH)



