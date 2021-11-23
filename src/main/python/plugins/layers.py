#! /usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import time
from tqdm import tqdm
from scipy.sparse import csr_matrix

DEBUG = False
"""
    The layers of our algorithm
"""
def Gauss2D(shape=(3,3),sigma=1):
    parameters = {'name': "Gauss2D",
                  'shape': shape,
                  'sigma': sigma}
    return parameters

def GaussDiff(shapes, sigmas):
    parameters = {'name': "GaussDiff",
                  'shapes': shapes,
                  'sigmas': sigmas}
    return parameters

def FeatExtract():
    parameters = {'name': "FeatExtract"}
    return parameters


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

    index = NeighborsIndex2D(n, m, width, height)
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

def GaussDiff_compile(data, gauss1, gauss2, param):
    if len(gauss1) != len(gauss2):
            err_message = "ERROR GaussDiff_compile: Gaussian not the same size"
            print(err_message)

    # Set progress bar
    total_event = len(gauss1)
    desc = 'GaussDiff...'
    pbar = tqdm(total=total_event, desc=desc)
    
    n, m, rgb = gauss1[0].shape
    low_height, low_width = param['low_shape']
    high_height, high_width = param['high_shape']
    index_low = NeighborsIndex2D(n, m, low_width, low_height)
    index_low = np.pad(np.reshape(index_low,(n-(low_height - 1),m-(low_width-1),low_height*low_height)), ((low_height // 2,low_height // 2),(low_width // 2,low_width // 2),(0,0)))
    index_high = NeighborsIndex2D(n, m, high_width, high_height)
    index_high = np.pad(np.reshape(index_high,(n-(high_height - 1),m-(high_width-1),high_height*high_height)), ((high_height // 2,high_height // 2),(high_width // 2,high_width // 2),(0,0)))
    new_mat = np.copy(data)
    new_mat = Grayscale2RGB(new_mat)
    for image in range(0, len(gauss1)):
        # Difference between gaussian
        diff_gauss = gauss1[image]-gauss2[image]
        diff_gauss_abs = np.absolute(diff_gauss)
        # Find maxima values horizontaly
        coord_x_horz = argrelextrema(diff_gauss_abs.flatten(), np.greater)[0] // np.array([n])
        coord_y_horz = argrelextrema(diff_gauss_abs.flatten(), np.greater)[0] % np.array([n])
        coord_horz = np.array((coord_x_horz,coord_y_horz)).T

        # Find maxima values verticaly
        coord_x_vert = argrelextrema(diff_gauss_abs.flatten(order='F'), np.greater)[0] % np.array([n])
        coord_y_vert = argrelextrema(diff_gauss_abs.flatten(order='F'), np.greater)[0] // np.array([n])
        coord_vert = np.array((coord_x_vert, coord_y_vert)).T
        coord = np.vstack([coord_horz,coord_vert])
        #coord = coord[np.lexsort((coord[:,1], coord[:,0]))]
        # Check 26 values around

        # Get unique coordinate at intersection
        unq, unq_num = np.unique(coord, return_counts=True, axis=0)
        coord_unq = unq[unq_num > 1]

        # Extract scaling
        scaling = ExtractScaling(coord_unq, diff_gauss, param['low_sigma'], param['high_sigma'])

        # Extract scaling
        orientation = ExtractOrientation(coord_unq, data[image], scaling, index_low, index_high)

        # Extract scaling
        appearance = ExtractAppearance(coord_unq)

        # Get matrix of keypoints
        new_mat_by_img = new_mat[image]
        new_mat_by_img[coord_unq[:,0],coord_unq[:,1],:] = np.array([255,0,0])
        new_mat[image] = new_mat_by_img
        pbar.update(1)
    pbar.close
    keypoints = 1
    return new_mat, keypoints

def FeatExtract_compile(data, param):
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

def NeighborsIndex2D(n, m, width, height):
    # Create matrix with index for each filters
    index = []
    index = np.concatenate([np.arange(i*m,i*m+width) for i in range(0, height)])
    index = np.tile(index, ((n-(height-1))*(m-(width-1)),1))

    # Make sure to skip edge with the size of the filter
    edge = np.concatenate([np.arange(0+(i*m),(m-(width-1))+(i*m)) for i in range(0,n-(height-1))])
    edge = np.tile(np.reshape(edge, ((n-(height-1))*(m-(width-1)),1)),(1,height*width))
    indexes = index + edge
    return indexes

def Grayscale2RGB(images_BGR):
    images_RGB = np.reshape(np.repeat(images_BGR, 3, axis=2),(len(images_BGR),256,256,3))
    return images_RGB

def ExtractScaling(xy_pos, diff_gauss, low_sigma, high_sigma):
    sign_keypoint = diff_gauss[xy_pos[:,0],xy_pos[:,1]] > 0
    scales = np.where(sign_keypoint > 0, high_sigma, low_sigma)
    return scales

def ExtractOrientation(xy_pos, data, scaling, index_low, index_high):
    # nb_bin = 9
    orientations = np.zeros((len(scaling),1))
    low_val = np.unique(scaling)[0]
    x_gradient, y_gradient = np.gradient(np.reshape(data,(256,256)))
    angles_pixel = np.arctan2(y_gradient, x_gradient).flatten()
    for idx in range(0,len(scaling)):
        if scaling[idx] == low_val:
            histogram_angle = np.histogram(angles_pixel[index_low[xy_pos[idx][0],xy_pos[idx][1],:]], nb_bin, weights=np.absolute(angles_pixel[index_low[xy_pos[idx][0],xy_pos[idx][1],:]]))
        else:
            histogram_angle = np.histogram(angles_pixel[index_high[xy_pos[idx][0],xy_pos[idx][1],:]], nb_bin, weights=np.absolute(angles_pixel[index_high[xy_pos[idx][0],xy_pos[idx][1],:]]))
        if (histogram_angle[0] == 0).all():
            orientations[idx] = np.array([0])
        else:
            idx_max = np.where(histogram_angle[0] == np.max(histogram_angle[0]))[0]
            orientation = np.array([(histogram_angle[1][idx_max[0]] + histogram_angle[1][idx_max[0]+1])/2])
            orientations[idx] = orientation

    # first_bin = (angles_pixel >= -np.pi/8) * (angles_pixel < np.pi/8)
    # sec_bin = (angles_pixel >= np.pi/8) * (angles_pixel < 3*np.pi/8)
    # third_bin = (angles_pixel >= 3*np.pi/8) * (angles_pixel < 5*np.pi/8)
    # fourth_bin = (angles_pixel >= 5*np.pi/8) * (angles_pixel < 7*np.pi/8)
    # fifth_bin = (angles_pixel >= 7*np.pi/8) + (angles_pixel < -7*np.pi/8)
    # sixth_bin = (angles_pixel >= -7*np.pi/8) * (angles_pixel < -5*np.pi/8)
    # seventh_bin = (angles_pixel >= -5*np.pi/8) * (angles_pixel < -3*np.pi/8)
    # height_bin = (angles_pixel >= -3*np.pi/8) * (angles_pixel < -np.pi/8)

    # histogram_angle = np.zeros(len(angles_pixel))
    # histogram_angle[np.where(first_bin)] = 0
    # histogram_angle[np.where(sec_bin)] = np.pi/4
    # histogram_angle[np.where(third_bin)] = np.pi/2
    # histogram_angle[np.where(fourth_bin)] = 3*np.pi/4
    # histogram_angle[np.where(fifth_bin)] = np.pi
    # histogram_angle[np.where(sixth_bin)] = 5*np.pi/4
    # histogram_angle[np.where(seventh_bin)] = 3*np.pi/2
    # histogram_angle[np.where(height_bin)] = 7*np.pi/4

    # for idx in range(0,len(scaling)):
    #     if scaling[idx] == low_val:
    #         angles, number = np.unique(histogram_angle[index_low[xy_pos[idx][0],xy_pos[idx][1],:]],return_counts=True)
    #     else:
    #         angles, number = np.unique(histogram_angle[index_high[xy_pos[idx][0],xy_pos[idx][1],:]],return_counts=True)
        
    #     orientations[idx] = angles[np.where(number == np.max(number))]

    return orientations

def ExtractAppearance(xy_pos):
    pass