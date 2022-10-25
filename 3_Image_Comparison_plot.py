import numpy as np
import math
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from glob import glob

#PSNR
def PSNR(img1, img2):
    mse = np.mean((img1 - img2)**2) #Mean square error
    if mse == 0:
        return 100
    PIXEL_MAX = 255
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr



#SSIM
def SSIM(image, imageNoise):
    mse = mean_squared_error(image, imageNoise)
    ssimImage = ssim(image, imageNoise) # data_range = imageNoise.max() - imageNoise.min()
    return ssimImage


SR_image_paths = sorted(glob("SR_4DImages_-10,10,*"))

ground_image = cv2.imread("imageComparison/HR.png",0)
fibre_image = cv2.imread("imageComparison/LR0.png",0)
norm_img = cv2.imread("imageComparison/normalisation.png",0)
interp_img = cv2.imread("imageComparison/interpolation.png",0)
#SR1_img  = cv2.imread("imageComparison/SR1_1.png",0)
#SR4_img = cv2.imread("imageComparison/SR4_1.png",0)


label = 'PSNR: {:.2f}, SSIM: {:.2f}'
fig, ax = plt.subplots(2,3)
#ax = axes.ravel()

psnrFibre = PSNR(fibre_image, ground_image)
ssimFibre = SSIM(fibre_image, ground_image)

psnrNorm = PSNR(norm_img, ground_image)
ssimNorm = SSIM(norm_img, ground_image)

psnrInterp = PSNR(interp_img, ground_image)
ssimInterp = SSIM(interp_img, ground_image)

#psnr1 = PSNR(SR1_img, ground_image)
#ssim1 = SSIM(SR1_img, ground_image)

#psnr4 = PSNR(SR4_img, ground_image)
#ssim4 = SSIM(SR4_img, ground_image)

ax[0,0].imshow(ground_image, cmap='gray')
ax[0,0].set_title("Original")
ax[0,0].axes.xaxis.set_ticks([])
ax[0,0].axes.yaxis.set_ticks([])

ax[0,1].imshow(fibre_image, cmap='gray')
ax[0,1].set_xlabel(label.format(psnrFibre,ssimFibre))
ax[0,1].set_title("Original")
ax[0,1].axes.xaxis.set_ticks([])
ax[0,1].axes.yaxis.set_ticks([])

ax[0,2].imshow(norm_img, cmap='gray')
ax[0,2].set_xlabel(label.format(psnrNorm,ssimNorm))
ax[0,2].set_title("Normalisation")
ax[0,2].axes.xaxis.set_ticks([])
ax[0,2].axes.yaxis.set_ticks([])



ax[1,0].imshow(interp_img, cmap='gray')
ax[1,0].set_xlabel(label.format(psnrInterp,ssimInterp))
ax[1,0].set_title("Linear Interpolation")
ax[1,0].axes.xaxis.set_ticks([])
ax[1,0].axes.yaxis.set_ticks([])

#ax[1,1].imshow(SR1_img, cmap='gray')
#ax[1,1].set_xlabel(label.format(psnr1,ssim1))
#ax[1,1].set_title("SISR")
#ax[1,1].axes.xaxis.set_ticks([])
#ax[1,1].axes.yaxis.set_ticks([])

#ax[1,2].imshow(SR4_img, cmap='gray')
#ax[1,2].set_xlabel(label.format(psnr4,ssim4))
#ax[1,2].set_title("MFSR")
#ax[1,2].axes.xaxis.set_ticks([])
#ax[1,2].axes.yaxis.set_ticks([])

plt.show()
