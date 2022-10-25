import cv2 as cv
import numpy as np
import math
from PIL import Image
from numpy.core.function_base import linspace


#------SETTINGS------#
rot_left = -10
rot_right = 10
rot_segments = 11

save_path = "./image_datasets/masks/"
#--------------------#


create_thresh300 = False
create_rotations = True




if(create_thresh300):
    light_filed = cv.imread("./image_datasets/originals/lightfield.tif",0)

    #Binarise image
    ret,thresh = cv.threshold(light_filed,51,255,cv.THRESH_BINARY)

    #Cut out larger 300x300 image for rotation
    thresh300 = np.zeros((300,300))

    height, width  = image.shape

    center = [math.floor(height/2), math.floor(width/2)]

    top = center[0] - 150
    bottom = center[0] + 150
    left = center[1] - 150
    right = center[1] + 150

    thresh300 = thresh[left:right,top:bottom]

    #cv.imshow("Thresh", thresh300)
    cv.imwrite(save_path + "thresh300.png", thresh300)





if(create_rotations):
    thresh300 = Image.open(save_path + "thresh300.png") #Need to load image with PIL

    Mask256 = np.zeros((256,256))

    for rot in linspace(rot_left, rot_right, rot_segments): #-10,10,11
        bundle_rot = thresh300.rotate(rot)
        
        height, width  = bundle_rot.size

        center = [math.floor(height/2), math.floor(width/2)]

        top = center[0] - 128
        bottom = center[0] + 128
        left = center[1] - 128
        right = center[1] + 128

        Mask256 = bundle_rot.crop((left,top,right,bottom))

        #Mask256.show("MAsk")

        Mask256.save(save_path + f'Mask256_{math.floor(rot)}.png', "PNG")






cv.waitKey(0)