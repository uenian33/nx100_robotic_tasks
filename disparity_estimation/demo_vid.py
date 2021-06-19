
from PIL import Image
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import time
import queue, threading, time
from matplotlib import pyplot as plt

sys.path.append('Disparity/utils/') # add relative path

from Disparity.Disparity import DisparityEstimationDL, DisparityEstimationTradition

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


# example of using DL pipeline for disparity estimation
"""
destimation = DisparityEstimationDL(
                model_file_name="Disparity/utils/STTR/sttr_light_sceneflow_pretrained_model.pth.tar",
                wb_model_file_name="Disparity/utils/WB/models/")
#"""
destimation = DisparityEstimationTradition(
                wb_model_file_name="Disparity/utils/WB/models/")
#"""
  
# define a video capture object
#vid = cv2.VideoCapture(0)
vid = VideoCapture(-1)

while(True):
    #print('straming')
      
    # Capture the video frame
    # by frame
    image = vid.read()
    #print(image.shape)
    #cv2.imshow("test",image)

    #plt.subplot(1, 2, 1)
    #plt.show(image)
    #time.sleep(100)

    height, width, channels = image.shape
    left_image = image[0:height, 0:int(width/2)] #this line crops
    right_image = image[0:height,int(width/2):width] #this line crops

    left_image = cv2.resize(left_image, dsize=(int(width), int(height*2)), interpolation=cv2.INTER_CUBIC)
    right_image = cv2.resize(right_image, dsize=(int(width), int(height*2)), interpolation=cv2.INTER_CUBIC)


    #disp_pred, occ_pred = destimation.inference(left_image, right_image, white_balance=True, denoise=False)
    disp_pred = destimation.inference(left_image, right_image, white_balance=True, denoise=True)

    # Display the resulting frame
    cv2.imshow('frame', disp_pred)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
#cv2.destroyAllWindows()

"""
plt.figure(5)
plt.imshow(occ_pred)
plt.show()
"""
