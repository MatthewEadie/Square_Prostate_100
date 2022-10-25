"""
    Created by: Matthew Eadie
    Date: 10/01/22

    Work based off RAMS multiframe super resolution 
"""

#import cv2
from cv2 import imread, imwrite, imshow, circle, resize
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from scipy import ndimage
from scipy.ndimage import shift, rotate, label
from skimage.transform import rescale
from skimage.feature import masked_register_translation
from scipy.interpolate import griddata
from skimage.measure import regionprops
from math import floor
from numpy.core.function_base import linspace


def load_original(image_path):
    """
    Function to read in original clipped images to allow for dividing into 256x256 size samples

    Parameters
    ----------
    base_path:
        path to image dataset folder
    dataset_number:
        name of original image 
    """
    image = imread(image_path)
    return image


#Resize using openCV library
#Interpolation used to resize
def resize_image(image, x_size, y_size):
    resized_image = resize(image, (x_size, y_size), interpolation = cv2.INTER_AREA)
    
    return resized_image


def div_original(base_path, dataset_count, original_image, train_set):
    #Get width and height of image
    width = original_image.shape[1]
    height = original_image.shape[0]

    #Calculate hoow many slices they can be divided into
    width_slices = floor(width / 256)
    height_slices = floor(height / 256)
    amount_slices = width_slices * height_slices

    count = 0 #Used to name slices

    #Loop over X and Y axis to crop 256x256 slices
    for x in range(width_slices):   
        for y in range(height_slices):
            #Square coordinates to crop
            xStart = x * 256    
            xStop = xStart + 256
            yStart = y * 256
            yStop = yStart + 256

            slice = original_image[yStart:yStop,xStart:xStop] #croped slice
            dataset_count = save_slice(base_path, dataset_count, slice, train_set) #Save slice into folder
            count += 1 #Incement count for next slice

    return dataset_count


def save_slice(base_path, dataset_count, image_slice, train_set):

    #Different naming conventions for training and testingg datasets
    if(train_set):
        save_path = base_path + "/imagedata_training" + str(dataset_count) 
    else:
        save_path = base_path + "/imagedata_testing" + str(dataset_count) 

    #If the directory doesn't exist make it
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    imwrite(save_path + "/HR.png", image_slice) #Write image into folder as HR
    dataset_count+=1 #increment dataset counter so HR images are overwritten

    return dataset_count


def AverageCores(img):
    # use a boolean condition to find where pixel values are > 0 (cores)
    blobs = img > 0.0
    #label connected regions that satisfy this condition
    labels, nlabels = label(blobs)
    #Get properties of each labeled region
    props = regionprops(labels, intensity_image = img)

    #Append mean intensity of each region to array
    means = []
    for region in props:
        means.append(region.mean_intensity)

    x = img.shape[0]
    y = img.shape[1]
    image = np.zeros((x,y))

    #Loop over each pixel
    #change core pixel values to mean of core
    for i in range(x):
        for j in range(y):
            if(img[i,j] > 0):
                image[i,j] = means[labels[i,j]-1]
    
    return image






    return X, Y



def dataset_stack_4D(dataset, segments):
    #Create empty arrays for X and Y
    X = np.empty((len(dataset),256,256,segments)) #X channels = number of rotations
    Y = np.empty((len(dataset),256,256,3)) #Y channels = 3

    for image_number,folder_path in tqdm(enumerate(dataset)):
        #Get path to LR (rotated fibre) images
        #Get path to HR image
        LRs = sorted(glob(folder_path+"/LR*.png"))
        HR = sorted(glob(folder_path+"/HR.png"))

        L = len(LRs) #Number of LR images
        #Empty LR and HR stack for each dataset
        LR_stack = np.empty((256,256,L))

        #Loop over each LR image appending them into stack
        for i,img in enumerate(LRs):
            LR_stack[:,:,i] = imread(img,0)/256 #256,256,Segments
            
        #Read HR image into HR stack (only need it once)
        HR_stack = imread(HR[0])/256 #256,256,3

        #Add dataset LR and HR stack into arrays containing all datasets
        X[image_number,:,:,:] = LR_stack
        Y[image_number,:,:,:] = HR_stack

    return X, Y



def dataset_stack_1D(dataset, segments):
    X = np.empty((len(dataset) * segments,256,256,3))
    Y = np.empty((len(dataset) * segments,256,256,3))

    image_count = 0

    for folder_path in tqdm(dataset):
        LRs = sorted(glob(folder_path+"/LR*.png")) #4 LRs
        HR = sorted(glob(folder_path+"/HR*.png")) #1 HR input 4 times


        for img in LRs:
            X[image_count,:,:,:] = imread(img)/256 #256,256,4
            Y[image_count,:,:,:] = imread(HR[0])/256 #256,256,4

            image_count += 1

    return X, Y


def save_stack(stack_save_dir, file_name, FourD_Stack):
    #If save directory doesn't exist create it
    if not os.path.isdir(stack_save_dir):
        os.mkdir(stack_save_dir)

    #Save stack of all datasets into folder as .npy
    np.save(os.path.join(stack_save_dir, file_name), FourD_Stack)
