"""
    Created by: Matthew Eadie
    Date: 10/01/22

    Work based off RAMS multiframe super resolution 
"""

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



load_datasets = True
test_model = False
reconstruct_dataset = True

if(load_datasets):
    #Load testing datasets
    X_test = np.load(os.path.join(path_datasets, "X_test4D.npy")) #(32,256,256,4) Images to run through model
    Y_test = np.load(os.path.join(path_datasets, "Y_test4D.npy")) #(32,256,256,4) HR images for comparison

    #Load trained model
    MF_UNet = load_model(model_fp,compile=False)
    MF_UNet.compile(optimizer='adam',loss='mse')

    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of Y_test: {Y_test.shape}")


#Function to test model before committing to reconstructing entire testing dataset
if(test_model):
    test_image = X_test[test_index:test_index+1]

    print(f"Shape of test_image: {test_image.shape}")

    X_pred = MF_UNet(test_image)

    fig, ax = plt.subplots(3)
    ax[0].imshow(X_test[test_index,:,:,0])
    ax[0].set_title('LR')

    ax[1].imshow(X_pred[0,:,:,0])
    ax[1].set_title('Prediction')

    ax[2].imshow(Y_test[test_index])
    ax[2].set_title('HR')
    
    plt.show()







if(reconstruct_dataset):
    #If directory to save SR images doesn't exist make it
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    start = time.time() #Start time to calculate time per image

    for index in range(floor(X_test.shape[0])):
        test_image = X_test[index:index+1,]
        X_pred = MF_UNet(test_image)



        plt.imsave(save_path + "/SR{}.png".format(index), X_pred[0,:,:,0], cmap='gray')
    
    end = time.time() #Endtime to calculate time per image
    print(f'Time taken: {end - start}seconds')
    print(f'Time per image: {(end - start)/X_test.shape[0]}seconds')