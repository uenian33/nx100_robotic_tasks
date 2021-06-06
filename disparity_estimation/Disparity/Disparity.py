"""### get pretrained model"""

"""### verify gpu is configured correctly"""

from PIL import Image
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import time

from WB.classes import WBsRGB as wb_srgb


from STTR.module.sttr import STTR
from STTR.dataset.preprocess import normalization, compute_left_occ_region
from STTR.utilities.misc import NestedTensor


def rectify(left, right, cam_params):
    """
    This function is used to rectify left and right images based on camera parameters
    to be added
    """
    return left, right

def simple_preprocess(left, right, WB_model=None, denoise=False, white_balance=False, reshape=True):
    """
    simple image preprocessing pipeline using denoise and white balance model
    """
    im_h, im_w, _ = left.shape
    #print(im_w, im_h)

    if reshape:
        r_w, r_h = int(im_w*0.4), int(im_h*0.4)
        left = cv2.resize(left, dsize=(r_w, r_h), interpolation=cv2.INTER_CUBIC)
        right = cv2.resize(right, dsize=(r_w, r_h), interpolation=cv2.INTER_CUBIC)


    if denoise:
        left = cv2.fastNlMeansDenoisingColored(left,None,10,10,7,21)
        right = cv2.fastNlMeansDenoisingColored(right,None,10,10,7,21)

    if white_balance:
        left = WB_model.correctImage(left) * 255 # white balance it
        right = WB_model.correctImage(right) * 255 # white balance it

        left= np.uint8(left)
        right= np.uint8(right)

    return left, right

class DisparityEstimationTradition(object):
    def __init__(self,
                kernel_size=3,
                numDisparities=96,
                blockSize=7,
                window_size=9,
                disp12MaxDiff=1,
                uniquenessRatio=16,
                speckleRange=2,
                wb_model_file_name="Disparity/utils/WB/models/"
                ):
        super(DisparityEstimationTradition, self).__init__()

        # Default parameters
        args = type('', (), {})() # create empty args
        args.kernel_size=3,
        args.numDisparities=96,
        args.blockSize=7,
        args.P1=8*3*window_size**2,
        args.P2=32*3*window_size**2,
        args.disp12MaxDiff=1,
        args.uniquenessRatio=16,
        args.speckleRange=2,
        self.args = args
        # Load the DeepLearning white balance model
        self.wbModel = wb_srgb.WBsRGB(gamut_mapping=2,upgraded=0, weight_pth=wb_model_file_name)



    def rectify_pairs(self, left, right, cam_params):
        """
        This function is used to rectify left and right images based on camera parameters
        to be added
        """
        return rectify(left, right, cam_params)

    def inference(self, left, right,  denoise=False, white_balance=False, reshape=True):
        left, right = simple_preprocess(left, right, self.wbModel, denoise, white_balance, reshape)

        kernel_size = self.args.kernel_size[0]
        print(kernel_size)
        smooth_left = cv2.GaussianBlur(left, (kernel_size,kernel_size), 1.5)
        smooth_right = cv2.GaussianBlur(right, (kernel_size, kernel_size), 1.5)

        left_matcher = cv2.StereoSGBM_create(
            numDisparities=self.args.numDisparities[0],
            blockSize=self.args.blockSize[0],
            P1=self.args.P1[0],
            P2=self.args.P2[0],
            disp12MaxDiff=self.args.disp12MaxDiff[0],
            uniquenessRatio=self.args.uniquenessRatio[0],
            speckleRange=self.args.speckleRange[0],
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

        """
        fig = plt.figure(figsize=(wls_image.shape[1]/DPI, wls_image.shape[0]/DPI), dpi=DPI, frameon=False);
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(wls_image, cmap='jet');
        plt.savefig(name)
        plt.close()
        create_combined_output(left, right, name)
        """

        return wls_image


class DisparityEstimationDL(object):
    """The deep learning + GPU Based disparity estimation
        better performance and relatively faster computation
    """
    def __init__(self, 
                channel_dim = 128,
                position_encoding='sine1d_rel',
                num_attn_layers=6,
                nheads=4,
                regression_head='ot',
                context_adjustment_layer='cal',
                cal_num_blocks=8,
                cal_feat_dim=16,
                cal_expansion_ratio=4, 
                model_file_name="STTR/sttr_light_sceneflow_pretrained_model.pth.tar",
                wb_model_file_name="WB/models/"):
        super(DisparityEstimationDL, self).__init__()
        
        # Default parameters
        args = type('', (), {})() # create empty args
        args.channel_dim = 128
        args.position_encoding='sine1d_rel'
        args.num_attn_layers=6
        args.nheads=4
        args.regression_head='ot'
        args.context_adjustment_layer='cal'
        args.cal_num_blocks=8
        args.cal_feat_dim=16
        args.cal_expansion_ratio=4

        self.args = args


        self.model = STTR(self.args).cuda().eval()

        # Load the pretrained model
        checkpoint = torch.load(model_file_name)
        pretrained_dict = checkpoint['state_dict']
        self.model.load_state_dict(pretrained_dict)
        print("Pre-trained model successfully loaded.")

        # Load the DeepLearning white balance model
        self.wbModel = wb_srgb.WBsRGB(gamut_mapping=2,upgraded=0, weight_pth=wb_model_file_name)



    def rectify_pairs(self, left, right, cam_params):
        """
        This function is used to rectify left and right images based on camera parameters
        to be added
        """
        return rectify(left, right, cam_params)

    def inference(self, left, right, denoise=False, white_balance=False, reshape=True):
        left, right = self.rectify_pairs(left, right, None)
        """### Preprocessing"""
        left, right = simple_preprocess(left, right, self.wbModel, denoise, white_balance, reshape)
 
        # normalize
        input_data = {'left': left, 'right':right,}
        input_data = normalization(**input_data)

        # donwsample attention by stride of 3
        h, w, _ = left.shape
        bs = 1

        downsample = 3
        col_offset = int(downsample / 2)
        row_offset = int(downsample / 2)
        sampled_cols = torch.arange(col_offset, w, downsample)[None,].expand(bs, -1).cuda()
        sampled_rows = torch.arange(row_offset, h, downsample)[None,].expand(bs, -1).cuda()

        # build NestedTensor
        input_data = NestedTensor(input_data['left'].cuda()[None,],input_data['right'].cuda()[None,])

        """### inference"""

        output = self.model(input_data)

        # set disparity of occ area to 0
        disp_pred = output['disp_pred'].data.cpu().numpy()[0]
        occ_pred = output['occ_pred'].data.cpu().numpy()[0] > 0.5
        disp_pred[occ_pred] = 0.0

        """
        # visualize predicted disparity and occlusion map
        plt.figure(4)
        plt.imshow(disp_pred)
        plt.figure(5)
        plt.imshow(occ_pred)
        plt.show()
        """

        return disp_pred, occ_pred

