# import necessary packages
import config.srresnet_config as config
from IO.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
from PIL import Image
import numpy as np
import shutil
import random
import PIL
import cv2
import os


# load image paths

imageHRPaths = list(paths.list_images(config.INPUT_HR_IMAGES))
#print(imagePaths)
#random.shuffle(imagePaths)
total = 0

print("[info] generating patches of training data")
for imageHRPath in imageHRPaths:

    hr_image = cv2.imread(imageHRPath)


    '''build 1x factor images
    (h, w) = image.shape[:2]
    scaled = np.array(Image.fromarray(image).resize((int(w * 4), int(h * 4)),
                                                     resample=PIL.Image.BICUBIC))
    scaledPath = os.path.sep.join([config.IMAGES, "{}.png".format((imagePath.split('/'))[-1].split('.')[0][:4])])
    cv2.imwrite(scaledPath, scaled)
    '''


    #
    # adjust border of image
    (h, w) = hr_image.shape[:2]
    dH = int((h % config.OUTPUT_DIM)/2.0)
    dW = int((w % config.OUTPUT_DIM)/2.0)

    hr_image = hr_image[dH:h - dH,
            dW: w-dW]
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # (rh, rw) = image.shape[:2]
    # print(rh, rw)

    # maintain a window to view all parts of an image
    for y in range(0, h-config.OUTPUT_DIM+1, config.OUTPUT_DIM):
        for x in range(0, w-config.OUTPUT_DIM+1, config.OUTPUT_DIM):
            target = hr_image[y:y+config.OUTPUT_DIM,
                     x:x+config.OUTPUT_DIM]

            # only need to update this part when changing upscale factor
            downsample_target = np.array(Image.fromarray(target).resize((config.OUTPUT_DIM//4, config.OUTPUT_DIM//4),
                                                     resample=PIL.Image.BICUBIC))

            scaled = np.array(Image.fromarray(downsample_target).resize((config.INPUT_DIM, config.INPUT_DIM),
                                                     resample=PIL.Image.BICUBIC))

            # write the images
            targetPath = os.path.sep.join([config.LABELS, "{}.png".format(total)])
            scaledPath = os.path.sep.join([config.IMAGES, "{}.png".format(total)])

            cv2.imwrite(targetPath, target)
            cv2.imwrite(scaledPath, scaled)

            total += 1
            if total % 10000 == 0:
                print("[info] generated {total} images already...".format(total=total))



print("[INFO] building HDF5 datasets...")
inputPaths = sorted(list(paths.list_images(config.IMAGES)))
outputPaths = sorted(list(paths.list_images(config.LABELS)))

inputWriter = HDF5DatasetWriter((len(inputPaths), config.INPUT_DIM, config.INPUT_DIM, 3), config.INPUTS_DB)
outputWriter = HDF5DatasetWriter((len(outputPaths), config.OUTPUT_DIM, config.OUTPUT_DIM, 3), config.OUTPUTS_DB)

for (inputPath, outputPath) in zip(inputPaths, outputPaths):
    inputImage = cv2.imread(inputPath)
    outputImage = cv2.imread(outputPath)

    inputWriter.add([inputImage], [-1])
    outputWriter.add([outputImage], [-1])

inputWriter.close()
outputWriter.close()
