import cv2
# from skimage.exposure import rescale_intensity
# from skimage.segmentation import slic
# from skimage.util import img_as_float
# from skimage import io
import numpy as np

FOLDER_NAME =  'augmented_image'
EXTENSION='.png'

#flip
def flip_image(image, dir):
    image = cv2.flip(image, dir)
    cv2.imwrite(FOLDER_NAME + '/flip-' + str(dir) + EXTENSION, image)

def invert_image(image, channel):
    image = (channel - image)
    cv2.imwrite(FOLDER_NAME + '/invert-' + str(channel) + EXTENSION, image)

def add_light(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i/255.0) ** inv_gamma) * 255
                    for i in np.arange(0,256)]).astype('uint8')

    image = cv2.LUT(image, table)

    if gamma >=1:
        cv2.imwrite(FOLDER_NAME + '/ligth-' + str(gamma) + EXTENSION, image)
    else:
        cv2.imwrite(FOLDER_NAME + '/dark-' + str(gamma) + EXTENSION, image)

def add_light_color(image, color, gamma=1.0):
    inv_gamma = 1.0 / gamma
    image = (color - image)
    table = np.array([((i/255.0) ** inv_gamma) * 255
                    for i in np.arange(0,256)]).astype('uint8')

    image = cv2.LUT(image, table)

    if gamma >=1:
        cv2.imwrite(FOLDER_NAME+'/ligth_color-' + str(gamma) + EXTENSION, image)
    else:
        cv2.imwrite(FOLDER_NAME+'/dark_color-' + str(gamma) + EXTENSION, image)

def saturation_image(image, saturation):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v

    image == cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite(FOLDER_NAME + '/saturation-' + str(saturation) + EXTENSION,image)

def hue_image(image, saturation):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)
    image[:, :, 2] = v

    image == cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite(FOLDER_NAME + '/hue-' + str(saturation) + EXTENSION,image)

def multiply_image(image, R,G,B):
    image = image*[R,G,B]
    cv2.imwrite(
        FOLDER_NAME+'/multiply-'+str(R)+'*'+str(G)+'*'+str(B)+'*'+ EXTENSION,
        image)

def gaussian_blur(image, blur):
    image = cv2.GaussianBlur(image, (5,5), blur)
    cv2.imwrite(FOLDER_NAME+'/gaussian_blur-'+str(blur)+EXTENSION, image)

def averageing_blur(image, shift):
    image = cv2.blur(image, (shift, shift))
    cv2.imwrite(FOLDER_NAME+'/averageing_blur-'+str(shift)+EXTENSION, image)

def median_blur(image, shift):
    image = cv2.medianBlur(image, shift)
    cv2.imwrite(FOLDER_NAME+'/median_blur-'+str(shift)+EXTENSION, image)

def bilateral_blur(image, d, color, space):
    image = cv2.bilateralFilter(image, d, color, space)
    cv2.imwrite(
        FOLDER_NAME+'/bi_blur-'+str(d)+'*'+str(color)+'*'+str(space)+EXTENSION,
        image)

def erosion_image(image, shift):
    kernel = np.ones((shift,shift),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    cv2.imwrite(FOLDER_NAME+'/erosion-'+str(shift)+EXTENSION, image)

def dilatation_image(image, shift):
    kernel = np.ones((shift,shift),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    cv2.imwrite(FOLDER_NAME+'/dilatation-'+str(shift)+EXTENSION, image)

def opening_image(image, shift):
    kernel = np.ones((shift,shift),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(FOLDER_NAME+'/opening-'+str(shift)+EXTENSION, image)

def closing_image(image, shift):
    kernel = np.ones((shift,shift),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(FOLDER_NAME+'/closing-'+str(shift)+EXTENSION, image)

def morphological_gradient_image(image, shift):
    kernel = np.ones((shift,shift),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite(
        FOLDER_NAME+'/morphological_gradient-'+str(shift)+EXTENSION,
        image)

def top_hat_image(image, shift):
    kernel = np.ones((shift,shift),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    cv2.imwrite(FOLDER_NAME+'/top_hat-'+str(shift)+EXTENSION, image)

def black_hat_image(image, shift):
    kernel = np.ones((shift,shift),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    cv2.imwrite(FOLDER_NAME+'/black_hat-'+str(shift)+EXTENSION, image)
