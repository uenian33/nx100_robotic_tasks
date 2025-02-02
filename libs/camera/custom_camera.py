from PIL import Image
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import time
import os
import datetime
from threading import Thread
from queue import Queue

from ..disparity_estimation.Disparity.Disparity import DisparityEstimationDL, DisparityEstimationTradition


class CustomCamera(object):
    """docstring for CustomCamera"""
    def __init__(self, mirror=False,
                       disp_model_pth=os.getcwd() + "/libs/disparity_estimation/Disparity/utils/STTR/sttr_light_sceneflow_pretrained_model.pth.tar",
                       wb_model_pth=os.getcwd() + "/libs/disparity_estimation/Disparity/utils/WB/models/"):
        super(CustomCamera, self).__init__()
        self.data = None
        self.cam = cv2.VideoCapture(0)

        self.WIDTH = 640
        self.HEIGHT = 480

        self.center_x = self.WIDTH / 2
        self.center_y = self.HEIGHT / 2
        self.touched_zoom = False

        self.image_queue = Queue()
        self.video_queue = Queue()

        self.scale = 1
        self.__setup()

        self.recording = False

        self.mirror = mirror
        #self.destimation = DisparityEstimationTradition(
        #        wb_model_file_name="disparity_estimation/Disparity/utils/WB/models/")
        self.destimation = DisparityEstimationDL(
                model_file_name=disp_model_pth,
                wb_model_file_name=wb_model_pth)

        # depth scale value, e.g. 105 pixel value in depth is equal to 36 cm by physical measurement
        self.depth_scale = 36 / 104.5

        self.init_params()

    def init_params(self):
        self.left_intrinsics =  [[1176.96626,    0.,       727.86896],
                                 [   0.,      1178.14294,  518.23675],
                                 [   0.,         0.,         1.     ]]
        self.right_intrinsics =  [[1192.4447,     0.,       695.88598,],
                                 [   0.,      1193.82608,  518.85838,],
                                 [   0.,         0.,         1.     ]]
        self.distortion_l =  [[ 0.10172,  0.03529 , 0.01247,  0.00165, -0.37752]]
        self.distortion_r =  [[ 0.12611, -0.09944 , 0.01237, -0.00181, -0.25298]]
        self.relative_rotation = [[ 0.99984,  0.00141,  0.01795,],
                                 [-0.00148,  0.99999 , 0.00384,],
                                 [-0.01795, -0.00387,  0.99983]]
        self.relative_translation = [[-61.49301],
                                     [ -0.75656],
                                     [  4.20757]]

        self.extrinsics = None

    def __setup(self):
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        time.sleep(2)

    def get_image_bundle(self, rw=600,rh=400):
        image = self.data
        while (image is None):
            image = self.data
        height, width, channels = image.shape
        left_image = image[0:height, 0:int(width/2)] #this line crops
        right_image = image[0:height,int(width/2):width] #this line crops

        left_image = cv2.resize(left_image, dsize=(rw, rh), interpolation=cv2.INTER_CUBIC)
        right_image = cv2.resize(right_image, dsize=(rw, rh), interpolation=cv2.INTER_CUBIC)

        try:
            disp_pred, occ_pred = self.destimation.inference(left_image, right_image, white_balance=True, denoise=False, reshape=False)
            print(disp_pred.shape)
        except:
            disp_pred = self.destimation.inference(left_image, right_image, white_balance=True, denoise=False, reshape=False)

        return left_image, disp_pred

    def stream(self):
        # streaming thread 
        def streaming():
            self.ret = True
            while self.ret:
                self.ret, np_image = self.cam.read()
                if np_image is None:
                    continue
                if self.mirror:
                    np_image = cv2.flip(np_image, 1)
                if self.touched_zoom:
                    np_image = self.__zoom(np_image, (self.center_x, self.center_y))
                else:
                    if not self.scale == 1:
                        np_image = self.__zoom(np_image)
                self.data = np_image
                k = cv2.waitKey(1)
                if k == ord('q'):
                    self.release()
                    break

        Thread(target=streaming).start()


    def save_picture(self):
        ret, img = self.cam.read()
        if ret:
            now = datetime.datetime.now()
            date = now.strftime('%Y%m%d')
            hour = now.strftime('%H%M%S')
            user_id = '00001'
            filename = './images/cvui_{}_{}_{}.png'.format(date, hour, user_id)
            cv2.imwrite(filename, img)
            self.image_queue.put_nowait(filename)


    def get_current_image(self):
        frame = self.data
        if frame is not None:
            cv2.imshow('SMS', frame)
            cv2.setMouseCallback('SMS', self.mouse_callback)
            key = cv2.waitKey(1)
            return frame
        else:
            return None

    def show(self):
        while True:
            frame = self.data
            if frame is not None:
                print(self.data.shape)
                cv2.imshow('SMS', frame)
                cv2.setMouseCallback('SMS', self.mouse_callback)
            key = cv2.waitKey(1)
            if key == ord('q'):
                # q : close
                self.release()
                cv2.destroyAllWindows()
                break

            elif key == ord('p'):
                # p : take picture and save image (image folder)
                self.save_picture()

    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.get_location(x, y)
            self.zoom_in()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.zoom_out()

"""
if __name__ == '__main__':
    cam = Camera(mirror=True)
    cam.stream()
    import time
    time.sleep(2)
    cam.get_current_image()
    #cam.show()
"""