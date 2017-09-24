import cv2
import copy
import numpy as np


class VisionAlgorithm:

    def __init__(self):
        self.image_raw = []
        self.image_to_display = []
        self.image_in_process = []

    IMG_WIDTH = 640
    IMG_HEIGHT = 480

    def load_image(self):
        
        self.image_raw = cv2.imread("test_images/test_image_17.jpg")
        self.img_original_width = self.image_raw.shape[1]
        self.image_ratio = self.img_original_width/self.image_raw.shape[0]
        self.img_original_height = int(self.img_original_width / self.image_ratio)
        image_resized = cv2.resize(self.image_raw, (self.IMG_WIDTH, self.IMG_HEIGHT))
        self.image_to_display = copy.copy(image_resized)
        for i in range(0, self.IMG_WIDTH):
            for j in range(0, int((5/8)*self.IMG_HEIGHT)):
                image_resized[j][i] = (0, 0, 0)
        self.image_in_process = image_resized

    @staticmethod
    def show_image(image):
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    def calculate_length(self, pointx, pointy):
        length = np.sqrt((pointx[0]-pointy[0])**2 + (pointx[1]-pointy[1])**2)
        return int(length)

    def analyse_image(self):
        img_w = self.IMG_WIDTH
        img_h = self.IMG_HEIGHT

        img_w_diag = self.img_original_width
        img_h_diag = self.img_original_height

        image_hsv = cv2.cvtColor(self.image_in_process, cv2.COLOR_BGR2HSV_FULL)
        image2analyse = image_hsv[:, :, 2]
        #szukam krawędzi za pomocą metody Canney'ego - progi eksperymentalne
        image2analyse = cv2.medianBlur(image2analyse, 7)

        var, linie = cv2.threshold(image2analyse, 220, 255, cv2.THRESH_BINARY)  #linie, by szukać wymiarów
        self.show_image(linie) #test


        image_temp = cv2.Canny(image2analyse, 80, 200) #40, 500
        image_temp = cv2.copyMakeBorder(image_temp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)  # biała ramka na masce
        mask = cv2.resize(image_temp, (img_w+2, img_h+2))
        image_temp = cv2.resize(image_temp, (img_w, img_h))
        cv2.floodFill(image_temp, mask, (int(img_w/2), int((2/3)*img_h)), 255)
        morph_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        image_temp = cv2.erode(image_temp, morph_element, iterations=2)
        image_temp = cv2.medianBlur(image_temp, 3)
        image_temp = cv2.dilate(image_temp, morph_element, iterations=1)

        var, mask = cv2.threshold(image_temp, 10, 255, cv2.THRESH_BINARY)  # maska: 1-droga 0-otoczenie
        im2, contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, 2)
        contours = contours[0]
        epsilon = 0.01*cv2.arcLength(contours, True)
        contours = cv2.approxPolyDP(contours, epsilon, True)
        cv2.polylines(self.image_to_display, [contours], 1, (0, 255, 0), 4)  # rysuje polygon wokol drogi
        self.show_image(self.image_to_display)
        # corners_of_image = np.array([[1, img_h], [img_w, img_h], [img_w, 1], [1, 1]], dtype="float32")
        corners_of_image = np.array([[1, img_h_diag], [img_w_diag, img_h_diag], [img_w_diag, 1], [1, 1]], dtype="float32")

        contours = np.array([contours[0][0], contours[1][0], contours[2][0], contours[3][0]], dtype="float32")
        contours, coeff_h, coeff_w = self.scale(contours)
        transform = cv2.getPerspectiveTransform(contours, corners_of_image)

        # img_w_res = 1.23*self.calculate_length(contours[0], contours[1])
        # height_of_square = (self.calculate_length(contours[1], contours[2]) + self.calculate_length(contours[0], contours[3]))/2
        # print(height_of_square)
        self.image_to_display = cv2.warpPerspective(self.image_raw, transform, (img_w_diag, int(img_h_diag)), borderMode=cv2.BORDER_CONSTANT, flags=cv2.INTER_NEAREST)  # przeksztalcenie perspektywy / droga w widoku z gory jako prostokat
        self.image_to_display = cv2.resize(self.image_to_display, (int(0.2*self.img_original_width), int(0.2*self.img_original_height)))


    def scale(self, contours):
        coeff_h = self.img_original_height/self.IMG_HEIGHT
        coeff_w = self.img_original_width/self.IMG_WIDTH
        contours[:,1] = contours[:,1] * coeff_w
        contours[:, 0] = contours[:, 0] * coeff_h
        return contours, coeff_h, coeff_w

    def line_dimensions(self, img):
        #idz po dolnych liniach i zlicz ile pikseli, moze wykres

