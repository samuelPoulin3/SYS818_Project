#! /usr/bin/env python3
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

DEBUG = False
"""
    The layers of our algorithm
"""
def Registration():
    parameters = {'name': "Registration"}
    return parameters

def Gauss2D(shape=(3,3),sigma=1):
    parameters = {'name': "Gauss2D",
                  'shape': shape,
                  'sigma': sigma}
    return parameters

def FeatExtract():
    parameters = {'name': "FeatExtract"}
    return parameters

def Conv2D(shape=(3,3), sigma=1):
    parameters = {'name': "Conv2D",
                  'shape': shape,
                  'sigma': sigma}
    return parameters

def MaxPooling2D(shape=(3,3), sigma=1):
    parameters = {'name': "MaxPooling2D",
                  'shape': shape,
                  'sigma': sigma}
    return parameters

def Registration_compile(data, param):
    # Set progress bar
    total_event = len(data)
    desc = 'Registration...'
    pbar = tqdm(total=total_event, desc=desc)
    image_source = data[0]
    n, m, rgb = image_source.shape
    image_source = np.reshape(image_source, (n, m))
    img_source_fft = np.fft.fft2(image_source)
    img_source_power = np.square(np.absolute(img_source_fft))

    registrations = []
    # Run filter on all images
    for image in data:
        image = np.reshape(image, (n, m))
        if (image == image_source).all():
            registrations.append(image)
            continue
        image_fft = np.fft.fft2(image)
        image_power = np.square(np.absolute(image_fft))
        new_image = image_power * 2
        new_image = np.fft.ifft2(new_image)
        registrations.append(new_image)
        pbar.update(1)
    registrations = np.reshape(np.array(registrations), (len(data), n, m, rgb))
    pbar.close

    return data

def Gauss2D_compile(data, param):
    # Set progress bar
    total_event = len(data)
    desc = 'Gauss2D...'
    pbar = tqdm(total=total_event, desc=desc)

    # Set up gaussian filter
    height, width = param['shape']
    n, m, rgb = data[0].shape
    filter = matlab_style_gauss2D(shape=(height,width),sigma=param['sigma'])
    filter_flat = filter.flatten()

    # Create matrix with index for each filters
    index = []
    index = np.concatenate([np.arange(i*m,i*m+width) for i in range(0, height)])
    index = np.tile(index, ((n-(height-1))*(m-(width-1)),1))

    # Make sure to skip edge with the size of the filter
    edge = np.concatenate([np.arange(0+(i*m),(m-(width-1))+(i*m)) for i in range(0,n-(height-1))])
    edge = np.tile(np.reshape(edge, ((n-(height-1))*(m-(width-1)),1)),(1,height*width))
    index = index + edge

    gaussian_mat = []

    # Run filter on all images
    for image in data:
        data_flat = image.flatten()

        # Sum each filter * data
        mat_conv = np.sum(np.prod([data_flat[index],filter_flat]),axis=1)
        mat_conv = np.reshape(mat_conv, ((n-(height-1)),(m-(width-1))))

        # Pad the edge to get 0 all around
        mat_conv = np.pad(mat_conv, ((height // 2,height // 2),(width // 2,width // 2)))
        gaussian_mat.append(mat_conv)
        pbar.update(1)
    gaussian_mat = np.reshape(np.array(gaussian_mat), (len(data), n, m, rgb))
    pbar.close
    return gaussian_mat

def FeatExtract_compile(data, param):
    pass

def Conv2D_compile(data, param):
    pass

def MaxPooling2D_compile(data, param):
    pass

def matlab_style_gauss2D(shape=(3,3),sigma=1):
    """
    https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h