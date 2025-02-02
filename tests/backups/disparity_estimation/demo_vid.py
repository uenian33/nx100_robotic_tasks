
from PIL import Image
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import time

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = Queue.Queue()
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
        except Queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

sys.path.append('Disparity/utils/') # add relative path

from Disparity.Disparity import DisparityEstimationDL, DisparityEstimationTradition


# example of using DL pipeline for disparity estimation
destimation = DisparityEstimationDL(
                model_file_name="Disparity/utils/STTR/sttr_light_sceneflow_pretrained_model.pth.tar",
                wb_model_file_name="Disparity/utils/WB/models/")
left_image_path='../images/test2/camL/'+str(0)+'.jpg'
right_image_path='../images/test2/camR/'+str(0)+'.jpg'

  
# define a video capture object
#vid = cv2.VideoCapture(0)
vid = VideoCapture(0)

while(True):
      
    # Capture the video frame
    # by frame
    ret, image = vid.read()

    height, width, channels = image.shape
    left_image = image[0:height, 0:int(width/2)] #this line crops
    right_image = image[0:height,int(width/2):width] #this line crops

    left_image = cv2.resize(left_image, dsize=(600, 400), interpolation=cv2.INTER_CUBIC)
    right_image = cv2.resize(right_image, dsize=(600, 400), interpolation=cv2.INTER_CUBIC)


    disp_pred, occ_pred = destimation.inference(left_image, right_image, white_balance=True, denoise=False)

    # Display the resulting frame
    cv2.imshow('frame', left_image)
      
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
