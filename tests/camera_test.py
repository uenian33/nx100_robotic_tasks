
import cv2
import time
import os
import datetime
from threading import Thread
from queue import Queue


class Camera:
    def __init__(self, mirror=False):
        self.data = None
        self.cam = cv2.VideoCapture(-1)

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

    def __setup(self):
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        time.sleep(2)


    def stream(self):
        # streaming thread 함수
        def streaming():
            self.ret = True
            while self.ret:
                self.ret, np_image = self.cam.read()
                print(np_image.shape)
                if np_image is None:
                    print('no image')
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


if __name__ == '__main__':
   
    cam = Camera(mirror=True)
    cam.stream()
    time.sleep(2)
    import time
    time.sleep(2)
    cam.get_current_image()
    #cam.show()
