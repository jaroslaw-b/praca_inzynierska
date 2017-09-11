import cv2
import numpy as np

class VisionAlgorithm:
    def __init__(self):
        self.image_temp = []
        self.image_temp1 = []
    def load_image(self):
        self.image_raw = cv2.imread("test_images/test_image_5.jpg")
        self.image_resized = cv2.resize(self.image_raw, (640, 480))
        for i in range(1,640):
            for j in range(1, 320):
                self.image_resized[j][i] = 0
    def show_image(self, image):
        cv2.imshow("image_raw", image)
        cv2.waitKey(0)
    def analyse_image(self):
        self.image_hsv = cv2.cvtColor(self.image_resized, cv2.COLOR_BGR2HSV_FULL)
        image2analyse = self.image_hsv[:,:,2]
        self.image_temp = cv2.Canny(image2analyse, 40, 500)
        self.image_temp = cv2.copyMakeBorder(self.image_temp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value = [255, 255, 255])
        
