"""
    Created by: Matthew Eadie
    Date: 10/01/22

    Work based off RAMS multiframe super resolution 
"""

import cv2
import utils.preprocessing as utils
import os
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
from math import floor
from numpy import zeros
from numpy.core.function_base import linspace
import math


#----------
# Settings

dataset_amount = 15 #Number of datasets
dataset_dir = "./image_datasets" #Location of image datasets
dataset_output_dir = "image_stacks"
threshold_scale = 1 #Change the scale of mask
mask_name = "thresh256_x" + str(threshold_scale)
originals_dir = dataset_dir + "/originals/"

img_resolution = 0.316
bundle_resolution = 0.32 #um/pixel

rot_left = -10
rot_right = 10
rot_segments = 11

create_thresh300 = False
create_rotations = False
Slice_Originals = True
Create_Overlays = True
Create_4DStack = True
Create_1DStack = False
#----------



# The lightfield image is threshold to isolate cores
# A 300x300 image is used to stop errors when rotating a 256x256 image
if(create_thresh300):
    light_filed = cv.imread("./image_datasets/originals/lightfield.tif",0)

    #Binarise image
    ret,thresh = cv.threshold(light_filed,51,255,cv2.THRESH_BINARY)

    #Cut out larger 300x300 image for rotation
    thresh300 = np.zeros((300,300))

    height, width  = image.shape
    center = [math.floor(height/2), math.floor(width/2)]

    top = center[0] - 150
    bottom = center[0] + 150
    left = center[1] - 150
    right = center[1] + 150

    thresh300 = thresh[left:right,top:bottom]
    cv.imwrite("./image_datasets/masks/thresh300.png", thresh300)




# To recreate rotating the fibre bundles rotated thresholds are made prior to overlap
# once the 300x300 image is rotated a 256x256 image is cropped out the middle and saved with rotation angle
if(create_rotations):
    thresh300 = Image.open(dataset_dir + "/masks/" + "thresh300.png") #Need to load image with PIL
    Mask256 = zeros((256,256))  #Empty array to store rotated mask

    for rot in linspace(rot_left, rot_right, rot_segments): #-10,10,11
        bundle_rot = thresh300.rotate(rot) #rotate large fibre bundle (300x300)
        
        #use the width and height to find centre
        height, width  = bundle_rot.size 
        center = [math.floor(height/2), math.floor(width/2)]

        #Coordinates for 256x256 crop
        top = center[0] - 128
        bottom = center[0] + 128
        left = center[1] - 128
        right = center[1] + 128

        #Crop the image using above coordinates
        Mask256 = bundle_rot.crop((left,top,right,bottom))

        #Save rotated 256x256 mask with angle in name
        Mask256.save(dataset_dir + "/masks/" + f'Mask256_{math.floor(rot)}.png', "PNG")


#Images are cut into 256x256 to match fibre bundle mask size
#This increases number of training images
#Smaller images train faster
if(Slice_Originals):
    #Split original images into training and testing datasets
    Train_images = sorted(glob(originals_dir + "Image*"))
    Test_images = sorted(glob(originals_dir + "TestImage*"))


    print("Creating training slices from originals")
    dataset_count = 0
    for image_path in tqdm(Train_images): #Loop over all training original images
        dataset_image = utils.load_original(image_path) #Load original image

        #divide originals into 256x256 greyscale slices
        #save slices on original image into new folder per slice 
        dataset_count = utils.div_original(dataset_dir, dataset_count, dataset_image, train_set = True)


    print("Creating testing slices from test originals")
    dataset_count = 0
    for image_path in tqdm(Test_images):
        dataset_image = utils.load_original(image_path) #Load original images

        #divide originals into 256x256 greyscale slices
        #save slices on original image into new folder per slice 
        dataset_count = utils.div_original(dataset_dir, dataset_count, dataset_image, train_set = False)

    
    
#Overlap fibre bundle masks with sliced HR images
if(Create_Overlays):
    print("Creating slice overlays")

    datasets = sorted(glob(dataset_dir +"/imagedata*")) #Get all datasets (including ones for testing)
    masks = sorted(glob(dataset_dir +"/masks/Mask256*")) #Get all rotated 256x256 masks

    for folder_path in tqdm(datasets):
        #Read in HR image of each dataset
        #Image needs to be adjusted to 8bit
        img_HR = cv2.imread(folder_path + "/HR.png",0)/256

        for c,mask_path in enumerate(masks): #Loop over each mask
            mask = cv2.imread(mask_path,0) #Read in rotated mask

            masked = img_HR * mask #Multiply image with binary mask

            bundle_ave = utils.AverageCores(masked) #Average cores so each one only contains one value

            cv2.imwrite(folder_path + f'/LR{c}.png', bundle_ave) #Save overlapped image into dataset folder




if (Create_4DStack):
    #Empty arrays to store different datasets
    #X - Low resolution fibre bundle images
    #Y - HR slices used as ground truth for comparison
    X_train = []; Y_train = [] #Datasets used fortraining model
    X_val = []; Y_val = [] #Datasets used for validation of training
    X_test = []; Y_test = [] #Datasets used for testing trained model

    #Path to training and testing datasets
    training_datasets = sorted(glob(dataset_dir +"/imagedata_training*"))
    testing_datasets = sorted(glob(dataset_dir +"/imagedata_testing*"))

    #Order for testing dataset important for stitching
    testing_datasets.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    #Create 4D stacks of training images
    print("Creating training 4D stacks")
    X_train_stack, Y_train_stack = utils.dataset_stack_4D(training_datasets, rot_segments)

    #Create 4D stacks of testing images
    print("Creating testing 4D stacks")
    X_test_stack, Y_test_stack = utils.dataset_stack_4D(testing_datasets, rot_segments)



    #Divide training datasets into training and validation
    dataset_no = X_train_stack.shape[0]
    validation_split = int((dataset_no / 10) * 2)
    training_split = int(validation_split + 1)

    X_val = X_train_stack[0:validation_split] #First 20% 
    Y_val = Y_train_stack[0:validation_split] #First 20% 

    X_train = X_train_stack[training_split:] #Remaining 80%
    Y_train = Y_train_stack[training_split:] #Remaining 80% 

    #Save datastacks
    utils.save_stack(dataset_output_dir, 'X_train4D.npy', X_train)
    utils.save_stack(dataset_output_dir, 'Y_train4D.npy', Y_train)

    utils.save_stack(dataset_output_dir, 'X_val4D.npy', X_val)
    utils.save_stack(dataset_output_dir, 'Y_val4D.npy', Y_val)

    utils.save_stack(dataset_output_dir, 'X_test4D.npy', X_test_stack)
    utils.save_stack(dataset_output_dir, 'Y_test4D.npy', Y_test_stack)





if (Create_1DStack):

    X_train = []; Y_train = []
    X_test = []; Y_test = []

    training_datasets = sorted(glob(dataset_dir +"/imagedata_training*"))
    testing_datasets = sorted(glob(dataset_dir +"/imagedata_testing*"))

    print("Creating training 1D stacks")
    X_train_stack, Y_train_stack = utils.dataset_stack_1D(training_datasets, rot_segments)

    print("Creating testing 1D stacks")
    X_test_stack, Y_test_stack = utils.dataset_stack_1D(testing_datasets, rot_segments)

    #Divide training datasets into training and validation
    dataset_no = X_train_stack.shape[0]
    validation_split = int((dataset_no / 10) * 2)
    training_split = int(validation_split + 1)

    X_val = X_train_stack[0:validation_split,:,:,:] #First 20% 
    Y_val = Y_train_stack[0:validation_split] #First 20% only 3 channels

    X_train = X_train_stack[training_split:,:,:,:] #Remaining 80%
    Y_train = Y_train_stack[training_split:] #Remaining 80% 


    print(f"X_train_stack shape: {X_train_stack.shape}")    #(88,256,256,4)
    print(f"Y_train_stack shape: {Y_train_stack.shape}")    #(88,256,256,4)

    print(f"X_test_stack shape: {X_test_stack.shape}")      #(32,256,256,4)
    print(f"Y_test_stack shape: {Y_test_stack.shape}")      #(32,256,256,4)

    #cv2.imshow("X_val", X_val[0])
    #cv2.imshow("Y_val", Y_val[0])

    #cv2.imshow("X_train", X_train[0])
    #cv2.imshow("Y_train", Y_train[0])

    utils.save_stack(dataset_output_dir, 'X_train1D.npy', X_train)
    utils.save_stack(dataset_output_dir, 'Y_train1D.npy', Y_train)

    utils.save_stack(dataset_output_dir, 'X_val1D.npy', X_val)
    utils.save_stack(dataset_output_dir, 'Y_val1D.npy', Y_val)

    utils.save_stack(dataset_output_dir, 'X_test1D.npy', X_test_stack)
    utils.save_stack(dataset_output_dir, 'Y_test1D.npy', Y_test_stack)



cv2.waitKey(0)
