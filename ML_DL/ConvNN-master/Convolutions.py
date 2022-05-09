import sys

import cv2
import numpy as np


# Grayscale Image
def processImage(image):
    image = cv2.imread(image)
    print('Original Dimensions : ',image.shape)
    width = 10
    height = 10
    dim = (width, height)
    # resize image
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    print('Resized Dimensions : ',resized_image.shape)
    image = cv2.cvtColor(src=resized_image, code=cv2.COLOR_BGR2GRAY)
    return image

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))
    
    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    print(xImgShape)
    yImgShape = image.shape[0]
    print(yImgShape)

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    print(xOutput)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    print(yOutput)
    output = np.zeros((xOutput, yOutput))
    print(output)

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output


if __name__ == '__main__':
    # Grayscale Image
    image = processImage('Image.jpeg')
    print(image.shape)

    # Edge Detection Kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Convolve and Save Output
    output = convolve2D(image, kernel, padding=1)
    cv2.imwrite('102DConvolved.jpg', output)
