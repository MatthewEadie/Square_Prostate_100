"""
    Created by: Matthew Eadie
    Date: 10/01/22

    Work based off RAMS multiframe super resolution 
"""

import numpy as np
import os
from utils.model import model_MFAE, model_SIAE, model_optimiser, model_loss
import tensorflow as tf
import datetime
import cv2
import time


os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)

#physical_devices = tf.config.list_physical_devices('GPU')
#try:
#  tf.config.experimental.set_memory_growth(physical_devices[0], True)
#except:
#  # Invalid device or cannot modify virtual devices once initialized.
#  pass


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=10240)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

"""
log_dir can save the training logical of the Network 
"""
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
earlyStop_callback = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=5)

#----------
# Settings
LR_Size = 256
Channels = 11 #Number of segments
batch_size = 4
epochs = 500
path_datasets = "image_stacks"
model_filepath = "MFAE_Models/MFUNet_trained" + str(epochs) + "_" + str(Channels)
#----------


load_dataset = True
load_model = True
train_model = True


#gpus = tf.config.list_physical_devices('GPU')
#if gpus:
#    # Currently, memory growth needs to be the same across GPUs
#    for gpu in gpus:
#      tf.config.experimental.set_memory_growth(gpu, True)
#    logical_gpus = tf.config.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


#Load datasets for model to use for training and validation
if(load_dataset):
    #Load validation dataset
    X_val_dataset = np.load(os.path.join(path_datasets, "X_val4D.npy")) #41,256,256,11
    Y_val_dataset = np.load(os.path.join(path_datasets, "Y_val4D.npy")) #41,256,256,3

    #Load training dataset
    X_train_dataset = np.load(os.path.join(path_datasets, "X_train4D.npy")) #165,256,256,11
    Y_train_dataset = np.load(os.path.join(path_datasets, "Y_train4D.npy")) #165,256,256,3

    
#Load model used for immage reconstruction
if(load_model):
    MF_UNet = model_MFAE(LR_Size, Channels) #Load model 
    MF_UNet.summary() #Print summary of model

    MF_UNet_optimiser = model_optimiser() #Build model optimiser

    loss_compile = model_loss #Load model loss 


if(train_model):
    #Compile model, must be done before training
    MF_UNet.compile(optimizer = MF_UNet_optimiser,
                    loss = loss_compile,
                    metrics = ['mae','mse']
                    )

    start = time.time()

    #Train model using X,Y training dataset and X,Y validation datasets
    history = MF_UNet.fit(x = X_train_dataset, #LR training dataset
                          y = Y_train_dataset, #HR training dataset
                          validation_data = (X_val_dataset, Y_val_dataset), #Validation datasets
                          batch_size = batch_size,  #Number of image to train on at once
                          epochs = epochs, #Maximum bumber of epochs to train
                          callbacks = [tensorboard_callback, earlyStop_callback], #Using early stop callback to avoid overtraining
                          verbose = 1, 
                          use_multiprocessing = True
                          )

    end = time.time() #Endtime to calculate time per image
    print(f'Time taken: {end - start}seconds')

    MF_UNet.save(model_filepath) #Save model to use on testing dataset


cv2.waitKey(0)