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


#------INTERPOLATE 256 MASK AVERAGED------#
def Interpolation(image):
    #Use interpolation to smooth image
    shrinkFactorContrastMask = 0.9
    lowerSet = 0.3
    upperSat = 99.7

    rows = image.shape[0]
    cols = image.shape[1]

    binaryCores = image > 0.0
    labels, nlabels = ndimage.label(binaryCores)


    props = regionprops(labels, intensity_image = image)
    
    xCoords = []
    yCoords = []
    #coreSizeVector = []
    values = []
    for region in props:
        centre_coords = np.around(region.centroid,0)
        xCoords.append(round(centre_coords[1],0))
        yCoords.append(round(centre_coords[0],0))

        values.append(region.mean_intensity)

    x = np.linspace(1,256,256)
    y = np.linspace(1,256,256)

    Xgrid, Ygrid = np.meshgrid(x,y)

    coords = np.vstack((xCoords, yCoords)).T

    delaunTri = Delaunay(coords)
    #plt.triplot(coords[:,0], coords[:,1], delaunTri.simplices)
    #plt.scatter(xCoords, yCoords, marker = "+")
    #plt.imshow(image)
    #plt.show()


    #gridzNear = griddata(coords, values, (Xgrid, Ygrid), method='nearest')
    gridzLin = griddata(coords, values, (Xgrid, Ygrid), method='linear')
    #gridzCube = griddata(coords, values, (Xgrid, Ygrid), method='cubic')

    #fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    #ax[0,0].set_title("Binary core image")
    #ax[0,0].imshow(image)

    #ax[0,1].set_title("Nearest")
    #ax[0,1].imshow(gridzNear, cmap = 'gray')

    #ax[1,0].set_title("Linear")
    #ax[1,0].imshow(gridzLin, cmap = 'gray')

    #ax[1,1].set_title("Cubic")
    #ax[1,1].imshow(gridzCube, cmap = 'gray')
    #plt.show()

    return gridzLin








testing_datasets = sorted(glob("./image_comparison/image_datasets/imagedata_testing*"))


save_dir = "./Test_Interpolation/"

if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

print("Creating testing slices from test originals")
for imageNo,image_path in enumerate(testing_datasets):
#Load in original images
#image_path = Test_images[0]
    image = cv2.imread(image_path + "/LR0.png")

    interp_img = Interpolation(image)

    cv2.imwrite(save_dir + f'TestImage{imageNo}.png', interp_img)

    imageNo += 1