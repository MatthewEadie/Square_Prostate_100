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

def kmean(img, K):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((ground_image.shape))
    return res2

def histo(img):
    # Calculate histogram without mask
    hist = cv2.calcHist([img],[0],None,[256],[0,255])
    #hist2 = cv2.calcHist([res2],[1],None,[256],[0,256])
    #hist3 = cv2.calcHist([res2],[2],None,[256],[0,256])
    return hist


ground_image = cv2.imread("Kmeans_Comparison/Cropped.png")
SR_img = cv2.imread("Kmeans_Comparison/Stitched.png")
clusters = 4

ground_cluster = kmean(ground_image, clusters)
SR_cluster = kmean(SR_img, clusters)

ground_hist = histo(ground_cluster)
SR_hist = histo(SR_cluster)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

ax1.imshow(ground_image)
ax1.set_title("Ground truth")
ax1.set_xticks([])
ax1.set_yticks([])

ax2.imshow(ground_cluster)
ax2.set_title("Clustered GT (K={})".format(clusters))
ax2.set_xticks([])
ax2.set_yticks([])

ax3.plot(ground_hist)
ax3.set_title("Histogram of pixel values (GT)")
#ax3.set_xlim([50,256])
#ax3.set_ylim([0,25000])

ax4.imshow(SR_img)
ax4.set_title("Super resolution")
ax4.set_xticks([])
ax4.set_yticks([])

ax5.imshow(SR_cluster)
ax5.set_title("Clustered SR (K={})".format(clusters))
ax5.set_xticks([])
ax5.set_yticks([])

ax6.plot(SR_hist)
ax6.set_title("Histogram of pixel values (SR)")
#ax6.set_xlim([50,256])
#ax6.set_ylim([0,25000])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()


