import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
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
    cv2.imwrite(FOLDER_NAME + '/saturation' + str(saturation) + EXTENSION,image)

def hue_image(image, saturation):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)
    image[:, :, 2] = v

    image == cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite(FOLDER_NAME + '/hue' + str(saturation) + EXTENSION,image)


image_file = 'consolidation.png'
image = cv2.imread(image_file)
print("Read")


flip_image(image, 0)#Horizontal
flip_image(image, 1)#Vertical
flip_image(image, -1)#Both

invert_image(image, 255)
invert_image(image, 200)
invert_image(image, 150)
invert_image(image, 100)
invert_image(image, 50)

add_light(image, 1.5)
add_light(image, 2.0)
add_light(image, 2.5)
add_light(image, 3.0)
add_light(image, 4.0)
add_light(image, 5.0)
add_light(image, 0.7)
add_light(image, 0.3)
add_light(image, 0.1)

add_light_color(image, 255 ,1.5)
add_light_color(image, 200 ,2.0)
add_light_color(image, 150 ,2.5)
add_light_color(image, 100 ,3.0)
add_light_color(image, 50 ,4.0)
add_light_color(image, 255 ,5.0)
add_light_color(image, 150 ,0.7)
add_light_color(image, 100 ,0.3)
add_light_color(image, 200 ,0.1)

saturation_image(image, 50)
saturation_image(image, 100)
saturation_image(image, 150)
saturation_image(image, 200)

hue_image(image, 50)
hue_image(image, 100)
hue_image(image, 150)
hue_image(image, 200)
