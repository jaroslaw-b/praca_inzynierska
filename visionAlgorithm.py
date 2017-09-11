import cv2


class VisionAlgorithm:

    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    
    def __init__(self):
        self.image_to_display = []
        self.image_in_process = []
        
    def load_image(self):
        
        image_raw = cv2.imread("test_images/test_image_5.jpg")
        image_resized = cv2.resize(image_raw, (self.IMG_WIDTH, self.IMG_HEIGHT))
        for i in range(1, self.IMG_WIDTH):
            for j in range(1, self.IMG_HEIGHT-180):
                image_resized[j][i] = (255, 255, 0)
        self.image_in_process = image_resized

    @staticmethod
    def show_image(image):
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        
    def analyse_image(self):
        img_w = self.IMG_WIDTH
        img_h = self.IMG_HEIGHT
        image_hsv = cv2.cvtColor(self.image_in_process, cv2.COLOR_BGR2HSV_FULL)
        image2analyse = image_hsv[:, :, 2]
        image_temp = cv2.Canny(image2analyse, 40, 500)
        image_temp = cv2.copyMakeBorder(image_temp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        mask = cv2.resize(image_temp, (img_w+2, img_h+2))
        image_temp = cv2.resize(image_temp, (img_w, img_h))
        cv2.floodFill(image_temp, mask, (320, int(img_w/2)), 255)
        morph_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        image_temp = cv2.erode(image_temp, morph_element, iterations=2)
        image_temp = cv2.dilate(image_temp, morph_element, iterations=2)
        var, image_temp = cv2.threshold(image_temp, 10, 255, cv2.THRESH_BINARY)
        image_road = self.image_in_process
        for i in range(0, img_w):
            for j in range(0, img_h):
                if image_temp[j][i] == 0:
                    image_road[j][i] = (255, 255, 0)
        self.image_to_display = image_road
