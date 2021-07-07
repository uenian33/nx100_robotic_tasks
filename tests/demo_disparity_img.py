import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from PIL import Image
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import time


from libs.disparity_estimation.Disparity.Disparity import DisparityEstimationDL, DisparityEstimationTradition


# example of using DL pipeline for disparity estimation
destimation = DisparityEstimationDL(
                model_file_name="Disparity/utils/STTR/sttr_light_sceneflow_pretrained_model.pth.tar",
                wb_model_file_name="Disparity/utils/WB/models/")
left_image_path='../images/test2/camL/'+str(0)+'.jpg'
right_image_path='../images/test2/camR/'+str(0)+'.jpg'
left_image = cv2.resize(cv2.imread(left_image_path, cv2.IMREAD_COLOR), dsize=(1200, 600), interpolation=cv2.INTER_CUBIC)
right_image = cv2.resize(cv2.imread(right_image_path, cv2.IMREAD_COLOR), dsize=(1200, 600), interpolation=cv2.INTER_CUBIC)



disp_pred, occ_pred = destimation.inference(left_image, right_image, white_balance=True, denoise=False)
plt.figure(3)
plt.imshow(disp_pred)
"""
plt.figure(5)
plt.imshow(occ_pred)
plt.show()
"""

# example of using traditional pipeline for disparity estimation
destimation = DisparityEstimationTradition(
                wb_model_file_name="Disparity/utils/WB/models/")

disp_pred = destimation.inference(left_image, right_image, white_balance=True, denoise=False)

plt.figure(4)
plt.imshow(disp_pred)
plt.show()