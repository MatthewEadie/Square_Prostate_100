import numpy as np
import math
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from glob import glob
from statistics import median

font = {'size' : 12}

plt.rc('font', **font)  # pass in the font dict as kwargs

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




PSNRs = []
SSIMs = np.zeros((11,21)) #Channels, SR images



SR_folder_paths = sorted(glob("MFAE_4DImages_*"),key=len)

for x,folder_path in enumerate(SR_folder_paths):

    SR_image_path = sorted(glob(folder_path + '/SR*.png'),key=len)

    for y,img_path in enumerate(SR_image_path):

        HR_img = cv2.imread(f'MFAE_HR/HR{y}.png',0)
        SR_img = cv2.imread(img_path,0)

        #PSNRs.append(PSNR(SR_img, ground_image))
        SSIMs[x,y] = SSIM(SR_img, HR_img)

#print(SSIMs)

SSIM_averages = np.zeros((11))
SSIM_mins = np.zeros((11))
SSIM_Maxs = np.zeros((11))
SSIM_std = np.zeros((11))

for i in range(11):
    SSIM_averages[i] = np.mean(SSIMs[i])
    SSIM_mins[i] = min(SSIMs[i])
    SSIM_Maxs[i] = max(SSIMs[i])
    SSIM_std[i] = np.std(SSIMs[i]) 

#print(SSIM_averages)

Channels = [2,3,4,5,6,7,8,9,10,11,15]

ytop = SSIM_Maxs - SSIM_averages
ybot = SSIM_averages - SSIM_mins

fig, ax = plt.subplots()
ax.errorbar(Channels,SSIM_averages, yerr=SSIM_std, marker='x', linestyle='-', capsize=2, elinewidth=1, markeredgewidth=1)
plt.ylim(0.4,1)
#plt.title('')
plt.grid(which='both', axis='y', linestyle='-.')
plt.xlabel("Number of channels", fontsize = 12)
plt.ylabel("SSIM", fontsize = 12)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.show()

cv2.waitKey(0)
