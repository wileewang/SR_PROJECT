# import the necessary packages
from config import srgan_config as config
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import argparse
import PIL
import cv2


# solve failed to get convolution algorithm
import tensorflow as tf
cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=cfg)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
# ap.add_argument("-b", "--baseline", required=True,
#                 help="path to baseline image")
ap.add_argument("-o", "--output", required=True,
                help="path to output image")
args = vars(ap.parse_args())

# load the pre-trained model
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# load the input image, then grab the dimensions of the input image
# and crop the image such that it tiles nicely
print("[INFO] generating image...")
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
w -= int(w % config.SCALE)
h -= int(h % config.SCALE)
image = image[0:h, 0:w]

# crop 56*56 field of image and clip them into a new large image.
# adjust border of image
(h, w) = image.shape[:2]
dH = int((h % config.INPUT_DIM) / 2.0)
dW = int((w % config.INPUT_DIM) / 2.0)

image = image[dH:h - dH,
        dW: w - dW]

(h, w) = image.shape[:2]

highH = h * config.SCALE
highW = w * config.SCALE
# allocate memory for the output image
output = np.zeros((highH,highW,3))


# slide a window from left-to-right and top-to-bottom
for y in range(0, int(h/config.INPUT_DIM)):
    for x in range(0, int(w/config.INPUT_DIM)):
        # crop the ROI from our image
        crop = image[y*config.INPUT_DIM:(y+1)*config.INPUT_DIM,
               x*config.INPUT_DIM:(x+1)*config.INPUT_DIM].astype("float32")

        P = model.predict(np.expand_dims(crop, axis=0))
        P = P.reshape((config.OUTPUT_DIM, config.OUTPUT_DIM, 3))
        output[y*config.OUTPUT_DIM:(y+1)*config.OUTPUT_DIM, x*config.OUTPUT_DIM:(x+1)*config.OUTPUT_DIM] = P

output = np.clip(output, 0, 255).astype("uint8")

# write the output image to disk
cv2.imwrite(args["output"], output)
