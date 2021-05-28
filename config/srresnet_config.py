# import the necessary packages
import os
# initialize the batch size and number of epochs for training
BATCH_SIZE = 2
NUM_EPOCHS = 20

# use part of dataset
RATIO_DATASET = 0.1

# define upscaling factor and input and output shape
INPUT_DIM = 256
OUTPUT_DIM = 256

SCALE = 1

LR_SHAPE = (INPUT_DIM, INPUT_DIM, 3)
HR_SHAPE = (int(INPUT_DIM * SCALE), int(INPUT_DIM * SCALE), 3)



# define the path to the input images we will be using to build
# training crops
INPUT_HR_IMAGES = "dataset/DIV2K_train_HR/"
INPUT_LR_IMAGES = "dataset/DIV2K_train_LR_bicubic/X1/"

# define the path to the temporary output directories
BASE_OUTPUT = "dataset/DIV2K_train_HR_TEMP"
IMAGES = os.path.sep.join([BASE_OUTPUT, "images"])
LABELS = os.path.sep.join([BASE_OUTPUT, "labels"])

# define the path to the HDF5 files
HDF5_OUTPUT = "dataset"
INPUTS_DB = os.path.sep.join([HDF5_OUTPUT, "inputs_{scale_factor}x.hdf5".format(scale_factor=SCALE)])
OUTPUTS_DB = os.path.sep.join([HDF5_OUTPUT, "outputs.hdf5"])

# define the path to the output model file and the plot file
TRAIN_OUTPUT = "output"
MODEL_PATH = os.path.sep.join([TRAIN_OUTPUT, "epoch_{NUM_EPOCHS}_dataset_{RATIO_DATASET}.srresnet".format(NUM_EPOCHS=NUM_EPOCHS, RATIO_DATASET=RATIO_DATASET)])
PLOT_PATH = os.path.sep.join([TRAIN_OUTPUT, "srresnet_plot.png"])




