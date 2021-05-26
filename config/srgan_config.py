# import the necessary packages
import os

# define the path to the input images we will be using to build
# training crops
INPUT_IMAGES = "dataset/urban100/"

# define the path to the temporary output directories
BASE_OUTPUT = "output"
IMAGES = os.path.sep.join([BASE_OUTPUT, "images"])
LABELS = os.path.sep.join([BASE_OUTPUT, "labels"])

# define the path to the HDF5 files
INPUTS_DB = os.path.sep.join([BASE_OUTPUT, "inputs.hdf5"])
OUTPUTS_DB = os.path.sep.join([BASE_OUTPUT, "outputs.hdf5"])

# define the path to the output model file and the plot file
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "srgan.model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "srgan_plot.png"])

# initialize the batch size and number of epochs for training
BATCH_SIZE = 2
NUM_EPOCHS = 10

# define upscaling factor and input and output shape
SCALE = 4
LR_SHAPE = (56, 56, 3)
HR_SHAPE = (224, 224, 3)

INPUT_DIM = 56
OUTPUT_DIM = 224
