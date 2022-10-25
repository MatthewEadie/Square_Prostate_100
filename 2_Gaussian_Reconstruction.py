from scipy import ndimage
from scipy.interpolate import griddata
from skimage.measure import regionprops
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math
from glob import glob



#------NORMALISE 256 MASK AVERAGED------#
def Normalisation(image):
    #Normalise an image using openCV
    downscallingFactor = 4
    sigmaGauss = 20

    lowsat = 0.3 #%
    uppersat = 99 #%

    if(sigmaGauss/downscallingFactor > 2):
        kernalDiamCalculated = round((sigmaGauss/downscallingFactor * 8 + 1))
        if(kernalDiamCalculated % 2 == 0):
            kernalDiamCalculated += 1
    else:
        kernalDiamCalculated = 2 * math.ceil(2 * sigmaGauss/downscallingFactor) + 1

    imageNormOutput = cv2.GaussianBlur(image, (kernalDiamCalculated,kernalDiamCalculated), sigmaGauss/downscallingFactor)
    
    #lowerBoundaryInput = np.percentile(imageNorm, lowsat)
    #lowerBoundaryOutput = 0
    #upperBoundayInput = np.percentile(imageNorm, uppersat)
    #upperBoundaryOutput = 255
    #imageNormOut = (imageNorm - lowerBoundaryInput) * ((upperBoundaryOutput - lowerBoundaryOutput) / (upperBoundayInput - lowerBoundaryInput)) + lowerBoundaryOutput

    imageNormOutput *= 2

    return imageNormOutput




testing_datasets = sorted(glob("./image_datasets/imagedata_testing*"))

imageNo = 0

save_dir = "./Test_Gaussian/"

print("Creating testing slices from test originals")
for image_path in testing_datasets:
#Load in original images
#image_path = Test_images[0]
    dataset_image = cv2.imread(image_path + "/LR0.png")

    norm_img = Normalisation(dataset_image)

    cv2.imwrite(save_dir + f'TestImage{imageNo}.png', norm_img)

    imageNo += 1


cv2.waitKey(0)