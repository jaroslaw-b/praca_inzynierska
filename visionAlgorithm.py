import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



class VisionAlgorithm:
    def __init__(self):
        self.image_raw = []
        self.mask = []
        self.image_raw_with_lines = []
        self.image_to_display = []
        self.image_in_process = []
        self.width_of_lines = ()
        self.preprocessed_image = []
        self.image_lines = []

    IMG_WIDTH = 1280
    IMG_HEIGHT = 960

    def load_image(self, nb_of_test_image):

        self.image_raw = cv2.imread("test_images/test_image_"+str(nb_of_test_image)+".jpg") #15, 14
        self.img_original_width = self.image_raw.shape[1]
        self.image_ratio = self.img_original_width / self.image_raw.shape[0]
        self.img_original_height = int(self.img_original_width / self.image_ratio)
        self.image_raw = cv2.resize(self.image_raw, (self.IMG_WIDTH, self.IMG_HEIGHT))
        image_resized = self.image_raw
        self.image_to_display = copy.copy(image_resized)
        for i in range(0, self.IMG_WIDTH):
            for j in range(0, int((0 / 8) * self.IMG_HEIGHT)):
                image_resized[j][i] = (0, 0, 0)
        self.image_in_process = image_resized

    @staticmethod
    def show_image(image):
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    def show_image2(self):
        im1 = cv2.resize(self.image_raw, (480, 640))
        im2 = cv2.resize(self.mask, (480, 640))
        im2 = np.dstack((im2, im2, im2))
        imstack = im1
        imstack = np.hstack((imstack, im2))
        self.show_image(imstack)


        # f, axarr = plt.subplots(2, 2)
        #
        # axarr[0, 0].imshow(self.image_raw)
        # axarr[0, 0].axis('off')
        # axarr[0, 1].imshow(self.mask)
        # axarr[0, 1].axis('off')
        # axarr[1, 0].imshow(self.image_raw)
        # axarr[1, 0].axis('off')
        # axarr[1, 1].imshow(self.image_to_display)
        # axarr[1, 1].axis('off')
        #
        # plt.tight_layout()
        # manager = plt.get_current_fig_manager()
        # manager.resize(*manager.window.maxsize())
        # plt.show()

    @staticmethod
    def calculate_length(pointx, pointy):
        length = np.sqrt((pointx[0] - pointy[0]) ** 2 + (pointx[1] - pointy[1]) ** 2)
        return int(length)

    def preprocessing(self):
        img_w = self.IMG_WIDTH
        img_h = self.IMG_HEIGHT


        image_hsv = cv2.cvtColor(self.image_in_process, cv2.COLOR_BGR2HSV_FULL)
        image2analyse = image_hsv[:, :, 2]
        # szukam krawędzi za pomocą metody Canny'ego - progi eksperymentalne
        image2analyse = cv2.medianBlur(image2analyse, 7)

        image2analyse = self.gamma_correction(image2analyse, 2)
        self.image_lines = image2analyse
        # self.show_image(image2analyse)  # test
        image_temp = cv2.Canny(image2analyse, 80, 200)  # 40, 500
        image_temp = cv2.copyMakeBorder(image_temp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)  # biała ramka na masce
        mask = cv2.resize(image_temp, (img_w + 2, img_h + 2))
        image_temp = cv2.resize(image_temp, (img_w, img_h))
        cv2.floodFill(image_temp, mask, (img_w // 2, int((2 / 3) * img_h)), 255)
        morph_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        image_temp = cv2.erode(image_temp, morph_element, iterations=2)
        image_temp = cv2.medianBlur(image_temp, 3)
        image_temp = cv2.dilate(image_temp, morph_element, iterations=1)

        var, mask = cv2.threshold(image_temp, 10, 255, cv2.THRESH_BINARY)  # maska: 1-droga 0-otoczenie
        self.preprocessed_image = mask

    def process_image2(self): #not
        image1 = copy.copy(self.image_to_display)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV_FULL)
        image = image1[:, :, 2]
        # image = self.gamma_correction(image, 1.2)
        canny_edges = cv2.Canny(image, 50, 100)
        # self.show_image(canny_edges)
        imshape = image.shape
        lower_left = [imshape[1] // 12, imshape[0]] # do dopracowania
        lower_right = [imshape[1] - imshape[1] // 12, imshape[0]]
        top_left = [imshape[1] // 2 - imshape[1] // 5, imshape[0] // 2 + imshape[0] // 5]
        top_right = [imshape[1] // 2 + imshape[1] // 5, imshape[0] // 2 + imshape[0] // 5]
        vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
        roi_image = self.region_of_interest(canny_edges, vertices)
        self.image_lines = self.region_of_interest(image, vertices) # do linii
        # self.show_image(self.image_lines)
        rho = 2
        theta = np.pi / 180
        # threshold is minimum number of intersections in a grid for candidate line to go to output
        threshold = 40
        min_line_len = 50
        max_line_gap = 250
        lines = cv2.HoughLinesP(roi_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        line_img = np.zeros((roi_image.shape[0], roi_image.shape[1]), dtype=np.uint8)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), 55, thickness=2)
        cv2.line(line_img, tuple(lower_left), tuple(lower_right), 55, thickness=3)
        cv2.line(line_img, tuple(top_left), tuple(top_right), 55, thickness=3)
        self.image_to_display = line_img
        # self.show_image(line_img)
        mask = cv2.resize(line_img, (imshape[1] + 2, imshape[0] + 2))
        cv2.floodFill(line_img, mask, ((top_left[0]+top_right[0])//2, (top_left[1]+lower_right[1])//2), 255)

        morph_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        ine_img = cv2.erode(line_img, morph_element, iterations=1)
        # line_img = cv2.medianBlur(line_img, 3)
        # line_img = cv2.dilate(line_img, morph_element, iterations=1)

        var, mask = cv2.threshold(line_img, 200, 255, cv2.THRESH_BINARY)  # maska: 1-droga 0-otoczenie
        self.image_to_display = mask
        self.mask = mask
        self.preprocessed_image = mask


    def region_of_interest(self, img, vertices): #not
        mask = np.zeros_like(img)
        ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def process_image(self):
        img_w_diag = self.img_original_width
        img_h_diag = self.img_original_height
        mask = self.preprocessed_image
        lines_threshhold = 250
        im2, contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, 2)
        contours = contours[0]
        epsilon = 0.01 * cv2.arcLength(contours, True)
        contours = cv2.approxPolyDP(contours, epsilon, True)
        cv2.polylines(self.image_to_display, [contours], 1, (0, 255, 0), 4)  # rysuje polygon wokol drogi
        # corners_of_image = np.array([[1, img_h], [img_w, img_h], [img_w, 1], [1, 1]], dtype="float32")
        corners_of_image = np.array([[1, img_h_diag], [img_w_diag, img_h_diag], [img_w_diag, 1], [1, 1]],
                                    dtype="float32")

        contours = np.array([contours[0][0], contours[1][0], contours[2][0], contours[3][0]], dtype="float32")
        var, linie = cv2.threshold(self.image_lines, lines_threshhold, 255, cv2.THRESH_BINARY)  # linie, by szukać wymiarów
        while self.line_dimensions(linie, contours) == -1:
            lines_threshhold -= 10
            if lines_threshhold < 140:
                break
        # self.show_image(linie)

        cv2.polylines(self.image_raw, np.int32([contours]), 1, [0, 255, 0], thickness=3)
        contours, coeff_h, coeff_w = self.scale(contours)
        transform = cv2.getPerspectiveTransform(contours, corners_of_image)
        self.image_to_display = cv2.warpPerspective(self.image_raw, transform, (img_w_diag, int(img_h_diag)),
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    flags=cv2.INTER_TAB_SIZE)  # przeksztalcenie perspektywy / droga w widoku z gory jako prostokat
        # self.image_to_display = cv2.resize(self.image_to_display,
        #                                    (int(0.2 * self.img_original_width), int(0.2 * self.img_original_height)))

    def search_damage(self):
        img_road = cv2.cvtColor(self.image_to_display, cv2.COLOR_BGR2HSV_FULL)
        img_road = img_road[:, :, 2]
        img_road = cv2.medianBlur(img_road, 5)
        # img_road = self.gamma_correction(img_road, 2)
        # self.show_image(img_road)
        roi_size = 15
        rois_h = img_road.shape[0]//roi_size
        rois_w = img_road.shape[1]//roi_size
        mean_mat = np.zeros((rois_h, rois_w))
        for i in range(rois_h):
            for j in range(rois_w):
                img_block = img_road[i:i+10, j:j+10]
                mean_mat[i][j] = img_block.mean()
        mean = img_road.mean()
        for i in range(len(mean_mat)):
            for j in range(len(mean_mat[0])):
                if mean_mat[i][j] > mean:
                    mean_mat[i][j] = 255 #odchylenie
        cv2.imshow("asd", mean_mat/255)
        print(mean)

    def search_damage2(self):
        img_road = cv2.cvtColor(self.image_to_display, cv2.COLOR_BGR2HSV_FULL)
        img_road = img_road[:, :, 2]
        # damages = cv2.bilateralFilter(img_road, 9, 30, 80)
        # self.show_image(damages)
        damages = cv2.Canny(img_road, 50, 250)
        damages = cv2.dilate(damages, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        # self.show_image(damages)
        # damages = cv2.morphologyEx(damages, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))




    def gamma_correction(self, image, gamma):
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                image[j,i] = int((image[j,i]/255)**gamma*255)
        return image

    def scale(self, contours):
        coeff_h = self.img_original_height / self.IMG_HEIGHT
        coeff_w = self.img_original_width / self.IMG_WIDTH
        contours[:, 1] = contours[:, 1] * coeff_w
        contours[:, 0] = contours[:, 0] * coeff_h
        return contours, coeff_h, coeff_w

    def line_dimensions(self, img, contours):
        try:
            med_low = int(contours[0][1] + contours[1][1]) // 2  # średnie wysokości górnej i dolnej krawędzi
            med_high = int(contours[2][1] + contours[3][1]) // 2
            low_line_size = 0
            high_line_size = 0
            ROWS_LINE_COUNTING = 3
            for j in range(ROWS_LINE_COUNTING):
                flag_low = 0
                flag_high = 0
                for i in range(0, img.shape[1]):
                    if img[med_high + j, i] == 255 and flag_high == 0:  # zapewnia przeliczenie szerokości JEDNEJ, pierwszej napotkanej linii
                        flag_high = 1
                    if flag_high == 1 and img[med_high + j, i] == 0:
                        flag_high = -1
                    if img[med_high + j, i] == 255 and flag_high == 1:
                        high_line_size = high_line_size + 1

                    if img[med_low - j, i] == 255 and flag_low == 0:
                        flag_low = 1
                    if flag_low == 1 and img[med_low - j, i] == 0:
                        flag_low = -1
                    if img[med_low - j, i] == 255 and flag_low == 1:
                        low_line_size = low_line_size + 1
            high_line_size = high_line_size / ROWS_LINE_COUNTING
            low_line_size = low_line_size / ROWS_LINE_COUNTING
            self.width_of_lines = (
                high_line_size, low_line_size)  # szerokość jednej linii 12cm w pikselach na górze i na dole obrazka
            low_line_length_px = self.calculate_length(contours[0], contours[1])  # dlugosc linii w pikselach
            high_line_length_px = self.calculate_length(contours[2], contours[3])
            low_line_length = low_line_length_px / low_line_size * 0.12  # przeskalowanie
            high_line_length = high_line_length_px / high_line_size * 0.12
            med_road_width = (low_line_length + high_line_length)/2
            if med_road_width > 4 or med_road_width < 1.5:
                raise ValueError
            print('Szerokość badanego pasa to:', med_road_width,'m')
            return med_road_width
        except (ZeroDivisionError, ValueError):
            return -1
