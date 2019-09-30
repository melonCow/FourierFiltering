#!/usr/bin/python3

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#######################################
## FILTER STUFF #######################

def apply_filter(image, filtr):

    """Applies the filter to the image."""

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    masked = np.multiply(filtr, fshift)

    fishift = np.fft.ifftshift(masked)

    return np.double(np.fft.ifft2(fishift))


def low_pass_filter(image, fsize):

    """Creates a low-pass square filter 
    (eliminates high frequencies)
    for the given image."""

    H, W = np.shape(img)
    mW = int(np.fix(0.5*W))
    mH = int(np.fix(0.5*H))   

    side = fsize
    mask = np.zeros((H,W))
    mask[mH-side:mH+side,mW-side:mW+side] = 1

    return mask

def gaussian_filter(image, sigma, reversed = False):

    H, W = np.shape(img)
    mW = int(np.fix(0.5*W))
    mH = int(np.fix(0.5*H))   

    mask = np.zeros((H,W))

    for i in range(H):
        for j in range(W):
            mask[i,j] = np.exp(-((i-mH)**2/(2.*sigma**2.) + (j-mW)**2./(2.* sigma**2.)))

    mask = mask/np.max(mask)

    if reversed:
        return 1 - mask
    else:
        return mask


def high_pass_filter(image, fsize):

    """Creates a high-pass square filter 
    (eliminates low frequencies)
    for the given image."""

    H, W = np.shape(img)
    mW = int(np.fix(0.5*W))
    mH = int(np.fix(0.5*H))   

    side = fsize
    mask = np.ones((H,W))
    mask[mH-side:mH+side,mW-side:mW+side] = 0

    return mask

def horizontal_filter(image, fsize, reversed=False):

    """Creates a horizontal band filter 
    for the given image."""

    H, W = np.shape(img)
    mH = int(np.fix(0.5*H))   

    if reversed:
        side = fsize
        mask = np.zeros((H,W))
        mask[mH-side:mH+side,:] = 1
    else:
        side = fsize
        mask = np.ones((H,W))
        mask[mH-side:mH+side,:] = 0

    return mask

def vertical_filter(image, fsize, reversed=False):

    """Creates a vertical band filter 
    for the given image."""

    H, W = np.shape(img)
    mW = int(np.fix(0.5*W))   

    if reversed:
        side = fsize
        mask = np.zeros((H,W))
        mask[:,mW-side:mW+side] = 1
    else:
        side = fsize
        mask = np.ones((H,W))
        mask[:,mW-side:mW+side] = 0

    return mask

#######################################

def plot_stuff(image, filtr, result):

    plt.figure(figsize=(900, 400))

    plt.subplot(1,3,1)
    plt.title("Image", fontsize=16)
    plt.imshow(image, cmap="gray")

    plt.subplot(1,3,2)
    plt.title("Filter", fontsize=16)
    plt.imshow(filtr, cmap="gray")

    plt.subplot(1,3,3)
    plt.title("Filtered Image", fontsize=16)
    plt.imshow(result, cmap="gray")

    plt.show()


## MAIN METHOD

if __name__ == "__main__":

    """ Launch this on execute."""

    src = "img/dog.jpg"
    img = np.array(Image.open(src).convert("L"))

    fsize = 30

    #mask = high_pass_filter(img, fsize)
    mask = gaussian_filter(img, 10, reversed=True)
    result = apply_filter(img, mask)

    plot_stuff(img, mask, result)

    print("")