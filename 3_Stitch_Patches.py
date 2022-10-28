import os
import cv2 as cv
import numpy as np
from glob import glob


#original = cv.imread("./image_datasets/originals_resized/TestImage0.tif")

#cropped = original[0:512,0:1024]

#cv.imwrite("Cropped.png", cropped)





xLength = 7
yLength = 3
slices = xLength * yLength


SR_path = "./Test_Interpolation/"

SR_imgs = os.listdir(SR_path)
#SR_imgs = sorted(glob(SR_path))

#SR_imgs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))



i = 0
j = 0
l = len(SR_imgs)

for z in range(int(l / slices)):

	stiched_img = np.zeros((yLength * 256,xLength * 256))


	for x in range(xLength):
		for y in range(yLength):
			img = cv.imread(SR_path + SR_imgs[i],0)


			left = x * 256
			right = left + 256
			top = y * 256
			bottom = top + 256


			stiched_img[top:bottom,left:right] = img


			i += 1

	cv.imwrite(SR_path + f'Stitched{j}.png', stiched_img)
	j += 1

#cv.imshow('Stitched', stiched_img/256)



cv.waitKey(0)
