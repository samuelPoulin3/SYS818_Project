#! /usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from PIL import Image
from itertools import compress
import warnings

warnings.filterwarnings("ignore")

DEBUG = False

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

def Resampling(images, size, scale):
    new_images = {}
    n, m = size
    new_images = csr_matrix(images.toarray()[0][np.reshape(np.arange(n*m),(n,m))[::scale,::scale].flatten()]).copy()
    return new_images

def NeighborsIndex2D(size, shape):
    # Create matrix with index for each filters
    index = []
    n, m = size
    height, width = shape

    # Get neighbors of first pixel (ex: n,m = 256,256 first pixel will be at (1,1) with
    # the index 0,1,2,256,257,258,512,513,514) 
    index = np.concatenate([np.arange(i*m,i*m+width) for i in range(0, height)])
    index = np.tile(index, ((n-(height-1))*(m-(width-1)),1))

    # The pixel that are too close to the side (shape // 2) will not be taken 
    # into account because we don't want pixel without all the neighbors
    edge = np.concatenate([np.arange(0+(i*m),(m-(width-1))+(i*m)) for i in range(0,n-(height-1))])
    edge = np.tile(np.reshape(edge, ((n-(height-1))*(m-(width-1)),1)),(1,height*width))
    indexes = index + edge
    return indexes

def Gauss_compute(image, neighbors_idx, filter_idx, size, shape, flag):
    # Sum each filter * data
    n, m = size
    height, width = shape

    # Check if it is first octave and perform gaussian otherwise keep same image
    if flag:
        # Multiply image by filter and sum all values
        mat_conv_1 = np.sum(np.prod([image.toarray().flatten()[neighbors_idx.toarray()],filter_idx[0]]),axis=1)

        # Reshape it so it will be easier to add zeros around to transform it back to original size
        mat_conv_1 = np.reshape(mat_conv_1, ((n-(2*(height // 2))),(m-(2*(width // 2)))))

        # Pad borders with 0
        mat_conv_1 = np.pad(mat_conv_1, ((height // 2,height // 2),(width // 2,width // 2)))

        # Put it back in sparse to take less space
        mat_conv_1 = csr_matrix(mat_conv_1.flatten())
    else:
        mat_conv_1 = image

    mat_conv_2 = np.sum(np.prod([mat_conv_1.toarray().flatten()[neighbors_idx.toarray()],filter_idx[1]]),axis=1)
    mat_conv_2 = np.reshape(mat_conv_2, ((n-(2*(height // 2))),(m-(2*(width // 2)))))
    mat_conv_2 = np.pad(mat_conv_2, ((height // 2,height // 2),(width // 2,width // 2)))
    mat_conv_2 = csr_matrix(mat_conv_2.flatten())

    dog_1 = np.absolute(mat_conv_1 - mat_conv_2)

    mat_conv_3 = np.sum(np.prod([mat_conv_2.toarray().flatten()[neighbors_idx.toarray()],filter_idx[2]]),axis=1)
    mat_conv_3 = np.reshape(mat_conv_3, ((n-(2*(height // 2))),(m-(2*(width // 2)))))
    mat_conv_3 = np.pad(mat_conv_3, ((height // 2,height // 2),(width // 2,width // 2)))
    mat_conv_3 = csr_matrix(mat_conv_3.flatten())

    dog_2 = np.absolute(mat_conv_2 - mat_conv_3)

    mat_conv_4 = np.sum(np.prod([mat_conv_3.toarray().flatten()[neighbors_idx.toarray()],filter_idx[3]]),axis=1)
    mat_conv_4 = np.reshape(mat_conv_4, ((n-(2*(height // 2))),(m-(2*(width // 2)))))
    mat_conv_4 = np.pad(mat_conv_4, ((height // 2,height // 2),(width // 2,width // 2)))
    mat_conv_4 = csr_matrix(mat_conv_4.flatten())

    dog_3 = np.absolute(mat_conv_3 - mat_conv_4)

    gaussian = [mat_conv_1,mat_conv_2,mat_conv_3,mat_conv_4]

    dogs = [dog_1,dog_2,dog_3]
    return gaussian, dogs

def using_complex(a):
    weight = 1j*np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
    b = a + weight[:, np.newaxis]
    u, ind = np.unique(b, return_index=True)
    b = np.zeros_like(a)
    np.put(b, ind, a.flat[ind])
    return b
    
class SIFT_Implementation():
    """
        The model of our keypoints
        The properties are:
            images: Dict of the images as sparse matrix 
    """

    __slots__ = ('n', 'm', 'key', 'images_1oct', 'images_2oct', 'images_3oct', 'sigma', \
                 'neighbors_2D_1', 'neighbors_2D_2', 'neighbors_2D_3', 'gaussian_filter', \
                 'gaussian_1', 'gaussian_2', 'gaussian_3', \
                 'dogs_1', 'dogs_2', 'dogs_3', 'threshold',\
                 'xy_coord', 'xy_coord_1', 'xy_coord_2', 'xy_coord_3',\
                 'gauss_1', 'gauss_2', 'gauss_3',\
                 'sigmas', 'sigmas_1', 'sigmas_2', 'sigmas_3',\
                 'magnitude', 'magnitude_1', 'magnitude_2', 'magnitude_3',\
                 'thetas', 'thetas_1', 'thetas_2', 'thetas_3',\
                 'appearance', 'appearance_1', 'appearance_2', 'appearance_3', 'keypoints')
    def __init__(self, images, key, og_size, *args, **kwargs):
        # General properties
        self.n = np.array([og_size[0], og_size[0] // 2, og_size[0] // 4])
        self.m = np.array([og_size[1], og_size[1] // 2, og_size[1] // 4])
        self.key = key

        # Get all images on each octave
        self.images_1oct = images
        self.images_2oct = {}
        self.images_3oct = {}

        # Gaussian related properties
        s = 3
        k = 2 ** (1/s)
        sig = 1.6
        width = 3
        height = 3

        # Establishing sigmas for all octaves
        self.sigma = np.array([[k * sig - sig, (k ** 2) * sig - (k * sig), (k ** 3) * sig - ((k ** 2) * sig), (k ** 4) * sig - ((k ** 3) * sig)],
                               [(k ** 4) * sig - ((k ** 3) * sig), (k ** 5) * sig - ((k ** 4) * sig), (k ** 6) * sig - ((k ** 5) * sig), (k ** 7) * sig - ((k ** 6) * sig)],
                               [(k ** 7) * sig - ((k ** 6) * sig), (k ** 8) * sig - ((k ** 7) * sig), (k ** 9) * sig - ((k ** 8) * sig), (k ** 10) * sig - ((k ** 9) * sig)]])

        # Get indexes of each octave gaussian
        self.neighbors_2D_1 = csr_matrix(NeighborsIndex2D((self.n[0],self.m[0]), (width, height)))
        self.neighbors_2D_2 = csr_matrix(NeighborsIndex2D((self.n[1],self.m[1]), (width, height)))
        self.neighbors_2D_3 = csr_matrix(NeighborsIndex2D((self.n[2],self.m[2]), (width, height)))

        # Get filter for each gaussian
        self.gaussian_filter = [matlab_style_gauss2D(shape=(width,height),sigma=self.sigma[i][j]).flatten() for i in range(self.sigma.shape[0]) for j in range(self.sigma.shape[1])]

        self.threshold = 0.03

        appearance_str = ["appearance_" + str(x) for x in range(1, 129)]
        self.keypoints = pd.DataFrame(columns= np.hstack(("image_no","keypoint_no", "x", "y", "sigma", "magnitude", "orientation", np.hstack(appearance_str))))
       
        self.SIFT_compute((width, height))
        self.get_keypoints()


    def SIFT_compute(self, shape):
        width, height = shape
        
        # Compute gaussians for each octave
        self.gaussian_1, self.dogs_1 = Gauss_compute(self.images_1oct, self.neighbors_2D_1, self.gaussian_filter[0:4], (self.n[0], self.m[0]), (width, height), True)
        self.images_2oct = Resampling(self.gaussian_1[3], (self.n[0], self.m[0]), 2)
        self.gaussian_2, self.dogs_2 = Gauss_compute(self.images_2oct, self.neighbors_2D_2, self.gaussian_filter[4:8], (self.n[1], self.m[1]), (width, height), False)
        self.images_3oct = Resampling(self.gaussian_2[3], (self.n[1], self.m[1]), 2)
        self.gaussian_3, self.dogs_3 = Gauss_compute(self.images_3oct, self.neighbors_2D_3, self.gaussian_filter[8:12], (self.n[2], self.m[2]), (width, height), False)

        # Compute xy coord for each octave
        self.xy_coord_1, self.gauss_1, self.sigmas_1 = self.XY_Extraction((self.n[0], self.m[0]), shape, self.dogs_1, self.neighbors_2D_1, self.sigma[0,1])
        self.xy_coord_2, self.gauss_2, self.sigmas_2 = self.XY_Extraction((self.n[1], self.m[1]), shape, self.dogs_2, self.neighbors_2D_2, self.sigma[1,1])
        self.xy_coord_3, self.gauss_3, self.sigmas_3 = self.XY_Extraction((self.n[2], self.m[2]), shape, self.dogs_3, self.neighbors_2D_3, self.sigma[2,1])

        # Delete heavy unused property
        self.neighbors_2D_1 = []
        self.neighbors_2D_2 = []
        self.neighbors_2D_3 = []

        self.dogs_1 = []
        self.dogs_2 = []
        self.dogs_3 = []

        # Compute orientation for all keypoints
        self.xy_coord_1, self.magnitude_1, self.thetas_1, self.appearance_1 = self.Orientation_Extraction(self.xy_coord_1, self.gaussian_1, self.gauss_1, self.sigma[0,1], (self.n[0], self.m[0]))
        self.xy_coord_2, self.magnitude_2, self.thetas_2, self.appearance_2 = self.Orientation_Extraction(self.xy_coord_2, self.gaussian_2, self.gauss_2, self.sigma[1,1], (self.n[1], self.m[1]))
        self.xy_coord_3, self.magnitude_3, self.thetas_3, self.appearance_3 = self.Orientation_Extraction(self.xy_coord_3, self.gaussian_3, self.gauss_3, self.sigma[2,1], (self.n[2], self.m[2]))

        sig_1 = np.full((len(self.xy_coord_1) - len(self.sigmas_1), ), self.sigmas_1[0])
        self.sigmas_1 = np.append(self.sigmas_1, sig_1)
        sig_2 = np.full((len(self.xy_coord_2) - len(self.sigmas_2), ), self.sigmas_2[0])
        self.sigmas_2 = np.append(self.sigmas_2, sig_2)
        sig_3 = np.full((len(self.xy_coord_3) - len(self.sigmas_3), ), self.sigmas_3[0])
        self.sigmas_3 = np.append(self.sigmas_3, sig_3)
        
    def XY_Extraction(self, size, shape, dogs, neighbors, sigma):
        n, m = size
        width, height = shape

        # Get all 27 values for each pixel available inside the borders
        neighbors_values = np.hstack([dogs[0].toarray()[0][neighbors.toarray()], 
                                      dogs[1].toarray()[0][neighbors.toarray()], 
                                      dogs[2].toarray()[0][neighbors.toarray()]])

        # Second derivatives
        d0  = np.gradient(np.reshape(dogs[0].toarray()[0],(n,m)))
        dxx_0 = np.gradient(d0[0], axis=0)
        dxy_0 = np.gradient(d0[0], axis=1)
        dyy_0 = np.gradient(d0[1], axis=1)
        
        d1  = np.gradient(np.reshape(dogs[1].toarray()[0],(n,m)))
        dxx_1 = np.gradient(d1[0], axis=0)
        dxy_1 = np.gradient(d1[0], axis=1)
        dyy_1 = np.gradient(d1[1], axis=1)

        d2  = np.gradient(np.reshape(dogs[2].toarray()[0],(n,m)))
        dxx_2 = np.gradient(d2[0], axis=0)
        dxy_2 = np.gradient(d2[0], axis=1)
        dyy_2 = np.gradient(d2[1], axis=1)

        # Check if maximum value of neighbors is in each pixel of our bottom, middle or top layer
        coord_max_top = np.argmax(neighbors_values,axis=1) == 4
        # Check which value is above threshold to delete low maximum
        coord_crit_top = neighbors_values[:,4] > self.threshold
        # Get values that respects both condition
        coord_top = coord_max_top * coord_crit_top
        coord_top = np.array(list(compress(range(len(coord_top)), coord_top))).astype(int)

        # Inspect gradient
        tr_0 = np.square(dxx_0.flatten()[coord_top] + dyy_0.flatten()[coord_top])
        det_0 = dxx_0.flatten()[coord_top] * dyy_0.flatten()[coord_top] - np.square(dxy_0.flatten()[coord_top])
        gradient_threshold_0 = np.nan_to_num(np.divide(tr_0,det_0))
        coord_top = np.delete(coord_top,np.where(gradient_threshold_0 >= np.array([np.square(11)/10])),0)
        gauss_top = np.full((len(coord_top), ), 0)

        coord_max_mid = np.argmax(neighbors_values,axis=1) == 13
        coord_crit_mid = neighbors_values[:,13] > self.threshold
        coord_mid = coord_max_mid * coord_crit_mid
        coord_mid = np.array(list(compress(range(len(coord_mid)), coord_mid))).astype(int)
        tr_1 = np.square(dxx_1.flatten()[coord_mid] + dyy_1.flatten()[coord_mid])
        det_1 = dxx_1.flatten()[coord_mid] * dyy_1.flatten()[coord_mid] - np.square(dxy_1.flatten()[coord_mid])
        gradient_threshold_1 = np.nan_to_num(np.divide(tr_1,det_1))
        coord_mid = np.delete(coord_mid,np.where(gradient_threshold_1 >= np.array([np.square(11)/10])),0)
        gauss_mid = np.full((len(coord_mid), ), 1)

        coord_max_bot = np.argmax(neighbors_values,axis=1) == 22
        coord_crit_bot = neighbors_values[:,22] > self.threshold
        coord_bot = coord_max_bot * coord_crit_bot
        coord_bot = np.array(list(compress(range(len(coord_bot)), coord_bot))).astype(int)
        tr_2 = np.square(dxx_2.flatten()[coord_bot] + dyy_2.flatten()[coord_bot])
        det_2 = dxx_2.flatten()[coord_bot] * dyy_2.flatten()[coord_bot] - np.square(dxy_2.flatten()[coord_bot])
        gradient_threshold_2 = np.nan_to_num(np.divide(tr_2,det_2))
        coord_bot = np.delete(coord_bot,np.where(gradient_threshold_2 >= np.array([np.square(11)/10])),0)
        gauss_bot = np.full((len(coord_bot), ), 2)

        gauss = np.hstack((gauss_top, gauss_mid, gauss_bot))
        coord = np.hstack((coord_top, coord_mid, coord_bot))

        if len(coord) != 0:
            coord = np.vstack(np.unravel_index(coord, (n - 2* (width // 2), m - 2* (height // 2)), order='C')).T.astype(int)
            # Add missing border to xy_pos
            coord = np.vstack([coord[:,0] + (width // 2), coord[:,1] + (height // 2)]).T
            # Delete some keypoints if too close to border before applying 16x16 filter for orientation
            gauss = np.delete(gauss,np.where(coord < ((16 // 2) + 1)),0)
            coord = np.delete(coord,np.where(coord < ((16 // 2) + 1)),0)
            gauss = np.delete(gauss,np.where(coord > (n - ((16 // 2) + 1))),0)
            coord = np.delete(coord,np.where(coord > (n - ((16 // 2) + 1))),0)

        sigmas = np.full((len(coord), 1), sigma)

        return coord, gauss, sigmas

    def Orientation_Extraction(self, xy_coord, gaussian, idx_gauss, sigma, size):
        n,m = size

        # Get indexes for 16x16 neighbors
        height_t, width_t = (16,16) 
        neighbors_orientation = NeighborsIndex2D((n, m), (height_t, width_t))

        # Adjust coordinates to be inside of the pixel available
        xy_coord_t = np.vstack([xy_coord[:,0] - height_t // 2, xy_coord[:,1] - width_t // 2]).T

        # Transform coord x,y into one coordinate
        key_index = neighbors_orientation[np.hstack(np.ravel_multi_index(np.hsplit(xy_coord_t,2),(n - (height_t - 1), m - (width_t - 1))))]
        key_num, index_num = key_index.shape

        key_index_xy = np.vstack(np.unravel_index(key_index.flatten(), (n, m), order='C')).T

        # Find each values to compute magnitude and orientation
        index_t = [np.ravel_multi_index((key_index_xy[:,0] + 1,key_index_xy[:,1]),(n,m)),\
                   np.ravel_multi_index((key_index_xy[:,0] - 1,key_index_xy[:,1]),(n,m)),\
                   np.ravel_multi_index((key_index_xy[:,0],key_index_xy[:,1] + 1),(n,m)),\
                   np.ravel_multi_index((key_index_xy[:,0],key_index_xy[:,1] - 1),(n,m))]

        # Get magnitudes
        idx_top = np.arange(len(np.where(idx_gauss == 0)[0])*index_num)
        idx_mid = np.arange(len(np.where(idx_gauss == 1)[0])*index_num)
        idx_bot = np.arange(len(np.where(idx_gauss == 2)[0])*index_num)
        magnitude_top = (((gaussian[0].T[index_t[0][idx_top]] - gaussian[0].T[index_t[1][idx_top]]).power(2)) + ((gaussian[0].T[index_t[2][idx_top]] - gaussian[0].T[index_t[3][idx_top]]).power(2))).sqrt()
        magnitude_mid = (((gaussian[1].T[index_t[0][idx_mid]] - gaussian[1].T[index_t[1][idx_mid]]).power(2)) + ((gaussian[1].T[index_t[2][idx_mid]] - gaussian[1].T[index_t[3][idx_mid]]).power(2))).sqrt()
        magnitude_bot = (((gaussian[2].T[index_t[0][idx_bot]] - gaussian[2].T[index_t[1][idx_bot]]).power(2)) + ((gaussian[2].T[index_t[2][idx_bot]] - gaussian[2].T[index_t[3][idx_bot]]).power(2))).sqrt()
        magnitude = vstack((magnitude_top,magnitude_mid,magnitude_bot))
        magnitude = np.reshape(magnitude,(key_num, index_num))
        filter = matlab_style_gauss2D(shape=(height_t,width_t),sigma=sigma*1.5).flatten()
        weight_magnitude = magnitude.multiply(filter).toarray()

        # Get orientation
        theta_top = np.arctan2((gaussian[0].T[index_t[2][idx_top]] - gaussian[0].T[index_t[3][idx_top]]).toarray(),(gaussian[0].T[index_t[0][idx_top]] - gaussian[0].T[index_t[1][idx_top]]).toarray())
        theta_mid = np.arctan2((gaussian[1].T[index_t[2][idx_mid]] - gaussian[1].T[index_t[3][idx_mid]]).toarray(),(gaussian[1].T[index_t[0][idx_mid]] - gaussian[1].T[index_t[1][idx_mid]]).toarray())
        theta_bot = np.arctan2((gaussian[2].T[index_t[2][idx_bot]] - gaussian[2].T[index_t[3][idx_bot]]).toarray(),(gaussian[2].T[index_t[0][idx_bot]] - gaussian[2].T[index_t[1][idx_bot]]).toarray())
        theta = np.vstack((theta_top, theta_mid, theta_bot))
        theta = (theta + 2*np.pi) * (theta < 0)
        theta = csr_matrix(theta)

        bins_theta = np.arange(0,2*np.pi,(2*np.pi)/35)
        theta_val = np.arange(0,360,360/36)
        theta = np.reshape(theta,(key_num, index_num))
        theta_histo = np.digitize(theta.toarray(),bins_theta,right=True)

        # build histogram
        index_histo = np.reshape(np.repeat(theta_histo,len(theta_val),axis=1),(key_num,index_num,len(theta_val))) == np.reshape(np.tile(np.arange(len(theta_val)),(key_num,index_num)), (key_num,index_num, len(theta_val)))
        value_histo = np.reshape(np.repeat(weight_magnitude,len(theta_val),axis=1),(key_num,index_num,len(theta_val))) * index_histo
        histogram = np.sum(value_histo,axis=1)

        # Get magnitudes and angles
        magnitudes = np.reshape(np.max(histogram,axis=1),(key_num,1))
        thetas = np.reshape(theta_val[np.argmax(histogram,axis=1)],(key_num,1))

        # Add keypoints with magnitude higher than 80%
        higher_val = histogram > np.reshape(np.tile(np.max(histogram,axis=1) *.8,len(theta_val)),(len(theta_val),key_num)).T
        nonmax_val = histogram != np.reshape(np.tile(np.max(histogram,axis=1),len(theta_val)),(len(theta_val),key_num)).T
        conditions = higher_val * nonmax_val
        add_key = np.where(conditions)

        # Append all new keypoints
        magnitudes = np.append(magnitudes, histogram[add_key[0],add_key[1]])
        thetas = np.append(thetas, theta_val[add_key[1]])
        xy_coord = np.append(xy_coord, xy_coord[add_key[0]],axis=0)

        # Get appearance
        theta_val_app = np.arange(0,360,360/8)
        bins_app = np.arange(0,2*np.pi,(2*np.pi)/7)
        index_app = np.reshape(np.arange(m),(16,m//16))
        orientation_app = theta.toarray()[:,index_app]
        weight_app = weight_magnitude[:,index_app]
        histo_app = np.digitize(orientation_app,bins_app,right=True)

        # build histogram
        index_histo_app = np.reshape(np.repeat(histo_app,len(theta_val_app),axis=2),(key_num,16,m//16,len(theta_val_app))) == np.reshape(np.tile(np.arange(len(theta_val_app)),(key_num,16,m//16)), (key_num, 16,m//16,len(theta_val_app)))
        value_histo_app = np.reshape(np.repeat(weight_app,len(theta_val_app),axis=2),(key_num,16,m//16,len(theta_val_app))) * index_histo_app
        histogram_app = np.sum(value_histo_app,axis=2)
        appearance = np.reshape(histogram_app,(key_num,128))

        appearance = np.append(appearance, appearance[add_key[0]],axis=0)

        return xy_coord, magnitudes, thetas, appearance

    def get_keypoints(self):
        self.xy_coord = np.vstack((self.xy_coord_1, self.xy_coord_2, self.xy_coord_3))
        charar = np.chararray((len(self.xy_coord), 1), itemsize=20)
        charar[:] = self.key
        keypoint_no = np.reshape(np.arange(0,len(self.xy_coord)),(len(self.xy_coord), 1))
        self.sigmas = np.reshape(np.hstack((self.sigmas_1, self.sigmas_2, self.sigmas_3)),(len(self.xy_coord),1))
        self.magnitude = np.reshape(np.hstack((self.magnitude_1, self.magnitude_2, self.magnitude_3)),(len(self.xy_coord),1))
        self.thetas = np.reshape(np.hstack((self.thetas_1, self.thetas_2, self.thetas_3)),(len(self.xy_coord),1))
        self.appearance = np.vstack((self.appearance_1, self.appearance_2, self.appearance_3))
        new_keypoints = np.hstack((charar, keypoint_no, self.xy_coord, self.sigmas, self.magnitude, self.thetas, self.appearance))
        appearance_str = ["appearance_" + str(x) for x in range(1, 129)]
        new_keypoints = pd.DataFrame(new_keypoints,columns= np.hstack(("image_no","keypoint_no", "x", "y", "sigma", "magnitude", "orientation", np.hstack(appearance_str))))
        self.keypoints = self.keypoints.append(new_keypoints, ignore_index=True)
        return self.keypoints

    def show_points(self, image2plot, points_x, points_y, shape):
        # self.show_points(self.images_1oct, np.hstack((self.xy_coord_1[:,0],self.xy_coord_2[:,0]*2,self.xy_coord_3[:,0]*4)),np.hstack((self.xy_coord_1[:,1],self.xy_coord_2[:,1]*2,self.xy_coord_3[:,1]*4)), (256,256))
        n, m = shape
        # Get matrix of keypoints
        image = np.reshape(image2plot.toarray(),(n,m,1))
        images_RGB = np.reshape(np.repeat(image, 3),(n*m,3))
        images_RGB = np.reshape(images_RGB,(n,m,3))
        if len(points_x) > 0 and len(points_y) > 0:
            images_RGB[points_x.astype(int),points_y.astype(int),:] = np.array([1,0,0])
        img = Image.fromarray(np.uint8(images_RGB * 255), "RGB")
        img.show()

    def show_images(self, image, size):
        # image as self.gaussian_1[key][0]
        n, m = size
        img = Image.fromarray(np.uint8(np.reshape(image.toarray().T * 255,(n,m))), "L")
        img.show()