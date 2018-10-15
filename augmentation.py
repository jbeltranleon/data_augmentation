import cv2
# from skimage.exposure import rescale_intensity
# from skimage.segmentation import slic
# from skimage.util import img_as_float
# from skimage import io
import numpy as np

class Booster:
    def __init__(self, folder, extension):
        self.folder = folder
        self.extension = extension

    #flip
    def flip_image(self, image, dir):
        image = cv2.flip(image, dir)
        cv2.imwrite(self.folder + '/flip-' + str(dir) + self.extension, image)

    def invert_image(self, image, channel):
        image = (channel - image)
        cv2.imwrite(self.folder + '/invert-' + str(channel) + self.extension, image)

    def add_light(self, image, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i/255.0) ** inv_gamma) * 255
                        for i in np.arange(0,256)]).astype('uint8')

        image = cv2.LUT(image, table)

        if gamma >=1:
            cv2.imwrite(self.folder + '/ligth-' + str(gamma) + self.extension, image)
        else:
            cv2.imwrite(self.folder + '/dark-' + str(gamma) + self.extension, image)

    def add_light_color(self, image, color, gamma=1.0):
        inv_gamma = 1.0 / gamma
        image = (color - image)
        table = np.array([((i/255.0) ** inv_gamma) * 255
                        for i in np.arange(0,256)]).astype('uint8')

        image = cv2.LUT(image, table)

        if gamma >=1:
            cv2.imwrite(self.folder+'/ligth_color-' + str(gamma) + self.extension, image)
        else:
            cv2.imwrite(self.folder+'/dark_color-' + str(gamma) + self.extension, image)

    def saturation_image(self, image, saturation):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        v = image[:, :, 2]
        v = np.where(v <= 255 - saturation, v + saturation, 255)
        image[:, :, 2] = v

        image == cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.imwrite(self.folder + '/saturation-' + str(saturation) + self.extension,image)

    def hue_image(self, image, saturation):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        v = image[:, :, 2]
        v = np.where(v <= 255 + saturation, v - saturation, 255)
        image[:, :, 2] = v

        image == cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.imwrite(self.folder + '/hue-' + str(saturation) + self.extension,image)

    def multiply_image(self, image, R,G,B):
        image = image*[R,G,B]
        cv2.imwrite(
            self.folder+'/multiply-'+str(R)+'*'+str(G)+'*'+str(B)+'*'+ self.extension,
            image)

    def gaussian_blur(self, image, blur):
        image = cv2.GaussianBlur(image, (5,5), blur)
        cv2.imwrite(self.folder+'/gaussian_blur-'+str(blur)+self.extension, image)

    def averageing_blur(self, image, shift):
        image = cv2.blur(image, (shift, shift))
        cv2.imwrite(self.folder+'/averageing_blur-'+str(shift)+self.extension, image)

    def median_blur(self, image, shift):
        image = cv2.medianBlur(image, shift)
        cv2.imwrite(self.folder+'/median_blur-'+str(shift)+self.extension, image)

    def bilateral_blur(self, image, d, color, space):
        image = cv2.bilateralFilter(image, d, color, space)
        cv2.imwrite(
            self.folder+'/bi_blur-'+str(d)+'*'+str(color)+'*'+str(space)+self.extension,
            image)

    def erosion_image(self, image, shift):
        kernel = np.ones((shift,shift),np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        cv2.imwrite(self.folder+'/erosion-'+str(shift)+self.extension, image)

    def dilatation_image(self, image, shift):
        kernel = np.ones((shift,shift),np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        cv2.imwrite(self.folder+'/dilatation-'+str(shift)+self.extension, image)

    def opening_image(self, image, shift):
        kernel = np.ones((shift,shift),np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(self.folder+'/opening-'+str(shift)+self.extension, image)

    def closing_image(self, image, shift):
        kernel = np.ones((shift,shift),np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(self.folder+'/closing-'+str(shift)+self.extension, image)

    def morphological_gradient_image(self, image, shift):
        kernel = np.ones((shift,shift),np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        cv2.imwrite(
            self.folder+'/morphological_gradient-'+str(shift)+self.extension,
            image)

    def top_hat_image(self, image, shift):
        kernel = np.ones((shift,shift),np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        cv2.imwrite(self.folder+'/top_hat-'+str(shift)+self.extension, image)

    def black_hat_image(self, image, shift):
        kernel = np.ones((shift,shift),np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        cv2.imwrite(self.folder+'/black_hat-'+str(shift)+self.extension, image)
