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
    ssimImage = ssim(image, imageNoise, multichannel=True) # data_range = imageNoise.max() - imageNoise.min()
    return ssimImage







file_path = glob('./image_comparison/MFAE_4DImages_11/*')
file_path.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

PSNR_ls = []
SSIM_ls = []



for x,filepath in enumerate(file_path):

    SR_image = cv2.imread(filepath)
    HR_image = cv2.imread(f'./image_comparison/image_datasets/imagedata_testing{x}/HR.png')

    psnrImg = PSNR(SR_image, HR_image)
    ssimImg = SSIM(SR_image, HR_image)

    PSNR_ls.append(psnrImg)
    SSIM_ls.append(ssimImg)


label = 'PSNR: {:.2f}, SSIM: {:.2f}'
#ax = axes.ravel()

plt.figure(1)
plt.plot(range(0,len(file_path)),PSNR_ls)

print('PSNR')
#mean
print(f'mean: {np.mean(PSNR_ls)}')
#max
print(f'maximum: {np.max(PSNR_ls)}')
#min
print(f'minimum: {np.min(PSNR_ls)}')
#std
print(f'standard div: {np.std(PSNR_ls)}')

plt.figure(2)
plt.plot(range(0,len(file_path)),SSIM_ls)

print()
print('SSIM')
#mean
print(f'mean: {np.mean(SSIM_ls)}')
#max
print(f'maximum: {np.max(SSIM_ls)}')
#min
print(f'minimum: {np.min(SSIM_ls)}')
#std
print(f'standard div: {np.std(SSIM_ls)}')

plt.show()
