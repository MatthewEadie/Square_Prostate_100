"""
    Created by: Matthew Eadie
    Date: 10/01/22

    Work based off RAMS multiframe super resolution 
"""

import numpy as np
import os
from utils.model import model_SIAE, model_optimiser, model_loss
import tensorflow as tf
import datetime
import cv2


"""
log_dir can save the training logical of the Network 
"""
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
earlyStop_callback = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=5)

#----------
# Settings
LR_Size = 256
Channels = 3
batch_size = 1
epochs = 500
path_datasets = "image_stacks"
model_filepath = "SIAE_trained" + str(epochs)
#----------


load_dataset = True
load_model = True
train_model = True

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


if(load_dataset):
    print("Loading datasets.")

    #Load training dataset
    X_train = np.load(os.path.join(path_datasets, "X_train1D.npy")) #(352,256,256,3)
    Y_train = np.load(os.path.join(path_datasets, "Y_train1D.npy")) #(352,256,256,3)

    X_val = np.load(os.path.join(path_datasets, "X_val1D.npy")) #(352,256,256,3)
    Y_val = np.load(os.path.join(path_datasets, "Y_val1D.npy")) #(352,256,256,3)


        
if(load_model):
    MF_UNet = model_SIAE(LR_Size, Channels)
    MF_UNet.summary()

    MF_UNet_optimiser = model_optimiser()

    loss_compile = model_loss


if(train_model):
    MF_UNet.compile(optimizer = MF_UNet_optimiser,
                    loss = loss_compile,
                    metrics = ['mae','mse']
                    )

    history = MF_UNet.fit(x = X_train,
                          y = Y_train,
                          validation_data = (X_val, Y_val),
                          batch_size = batch_size,
                          epochs = epochs,
                          callbacks = [tensorboard_callback, earlyStop_callback], 
                          verbose = 1, 
                          use_multiprocessing = True
                          )

    MF_UNet.save(model_filepath)


cv2.waitKey(0)
