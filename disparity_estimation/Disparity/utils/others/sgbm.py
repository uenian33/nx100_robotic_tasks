#!/usr/bin/python

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn.preprocessing import normalize
from WB.classes import WBsRGB as wb_srgb

DPI=96
DATASET = "data/1"
DATASET_LEFT = DATASET+"/left/"
DATASET_RIGHT = DATASET+"/right/"
DATASET_DISPARITIES = DATASET+"/disparities/"
DATASET_COMBINED = DATASET+"/combined/"

def process_frame(left, right, name):
    kernel_size = 3
    smooth_left = cv2.GaussianBlur(left, (kernel_size,kernel_size), 1.5)
    smooth_right = cv2.GaussianBlur(right, (kernel_size, kernel_size), 1.5)

    window_size = 9    
    left_matcher = cv2.StereoSGBM_create(
        numDisparities=96,
        blockSize=7,
        P1=8*3*window_size**2,
        P2=32*3*window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=16,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.2)

    disparity_left = np.int16(left_matcher.compute(smooth_left, smooth_right))
    disparity_right = np.int16(right_matcher.compute(smooth_right, smooth_left) )

    wls_image = wls_filter.filter(disparity_left, smooth_left, None, disparity_right)
    #wls_image = cv2.normalize(src=wls_image, dst=wls_image, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    wls_image = np.uint8(wls_image)

    fig = plt.figure(figsize=(wls_image.shape[1]/DPI, wls_image.shape[0]/DPI), dpi=DPI, frameon=False);
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(wls_image, cmap='jet');
    plt.savefig(name)
    plt.close()
    create_combined_output(left, right, name)

def create_combined_output(left, right, name):
    combined = np.concatenate((left, right, cv2.imread(name)), axis=0)
    cv2.imwrite(DATASET_COMBINED+name, combined)

def process_dataset():
    DATASET_LEFT = '../images/test/camL/'
    left_images = [f for f in os.listdir(DATASET_LEFT) if not f.startswith('.')]

    for i in range(70):
        left_image_path='../images/test/camL/'+str(i)+'.jpg'
        right_image_path='../images/test/camR/'+str(i)+'.jpg'
        left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
        right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)

        # processing
        # create an instance of the WB model
       

        left_image = cv2.fastNlMeansDenoisingColored(left_image,None,10,10,7,21)
        right_image = cv2.fastNlMeansDenoisingColored(right_image,None,10,10,7,21)

        print(left_image.shape, right_image.shape)

        wbModel = wb_srgb.WBsRGB(gamut_mapping=2,upgraded=0, weight_pth='WB/models/')
        left_image = wbModel.correctImage(left_image) * 255 # white balance it
        right_image = wbModel.correctImage(right_image) * 255 # white balance it

        left_image = np.uint8(left_image)
        right_image = np.uint8(right_image)
        print(left_image.shape, right_image.shape)

        process_frame(left_image, right_image, left_images[i])


def naive_sgbm():
    for i in range(70):
        print(i)
        left_image_path='../images/test2/camL/'+str(i)+'.jpg'
        right_image_path='../images/test2/camR/'+str(i)+'.jpg'
        imL=cv2.imread(left_image_path,0).astype('float32')/255.0
        imR=cv2.imread(right_image_path,0).astype('float32')/255.0
        

        k=31 # no of disparity levels
        B=5# Patch size of 2B+1 x 2B+1

        D      = np.ones( list( imL.shape )+[ k ] )
        d_list = [ int( x ) for x in np.linspace( 0, k-1, k ) ]

        for d in d_list:
            sq_diff = np.square( imR[ :, d: ] - imL[ :, 0:imL.shape[ 1 ] - d ] )
            result  = np.zeros_like(sq_diff)
            padded  = np.pad( sq_diff, B, mode = 'constant' )
            for r in range( B, padded.shape[ 0 ] - B ):
                for c in range( B, padded.shape[ 1 ] - B):       
                    vals = [ padded[ rr ][ cc ] for rr in range( r-B, r+B+1 ) for cc in range( c-B, c+B+1 ) ]
                    result[ r-B, c-B ] = np.mean( vals )   
            D[ :, 0:imL.shape[ 1 ] - d, d ] = result
            
        dm_est    = np.argmin( D, axis = 2 )
        dm_est_img= ( dm_est*8 ).astype( np.uint8 )
        cv2.imwrite(str(i)+'.jpg', dm_est_img)
                

if __name__== "__main__":
    left_image = cv2.imread("WB/result0.jpg", cv2.IMREAD_COLOR)
    right_image = cv2.imread("WB/result.jpg", cv2.IMREAD_COLOR)

    process_frame(left_image, right_image, "test.png")
    process_dataset()
    #naive_sgbm()