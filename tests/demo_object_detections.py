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
import queue, threading, time
from matplotlib import pyplot as plt


from libs.disparity_estimation.Disparity.Disparity import DisparityEstimationDL, DisparityEstimationTradition
from libs.object_detection.yoloDetector import ObjDetector 
print(cv2.__version__)
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


  
# define a video capture object
vid = cv2.VideoCapture(0)
#vid = VideoCapture(-1)

# define object detection pbject
obj_detector = ObjDetector()

while(True):
   
    ret, image = vid.read()
    print(image.shape)
   

    height, width, channels = image.shape
    left_image = image[0:height, 0:int(width/2)] #this line crops
    right_image = image[0:height,int(width/2):width] #this line crops

    left_image = cv2.resize(left_image, dsize=(int(width), int(height*2)), interpolation=cv2.INTER_CUBIC)
    right_image = cv2.resize(right_image, dsize=(int(width), int(height*2)), interpolation=cv2.INTER_CUBIC)


    detections = obj_detector(left_image)
    print(detections)
   
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
#cv2.destroyAllWindows()


