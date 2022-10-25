import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from cv2 import imwrite
from math import floor
import time

#----------
# Settings
channels = 15
model_fp = "MFAE_Models/MFUNet_trained500_" + str(channels) 
test_index = 3 #index of image to test model on
path_datasets = "image_stacks"
save_path = "MFAE_4DImages/MFAE_4DImages_" + str(channels) 
#----------

X_train = np.load(os.path.join(path_datasets, "X_train4D.npy")) #(32,256,256,4) Images to run through model
X_val = np.load(os.path.join(path_datasets, "X_val4D.npy")) #(32,256,256,4) Images to run through model
X_test = np.load(os.path.join(path_datasets, "X_test4D.npy")) #(32,256,256,4) Images to run through model


print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")
print(f"Shape of X_test: {X_test.shape}")