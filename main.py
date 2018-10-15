import cv2
from augmentation import *
from datetime import datetime

print('Inicio del Proceso')
start = datetime.now()

image_file = 'consolidation.png'
image = cv2.imread(image_file)
print("Read Complete")

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

multiply_image(image, 0.5, 1, 1)
multiply_image(image, 1, 0.5, 1)
multiply_image(image, 1, 1, 0.5)
multiply_image(image, 0.5, 0.5, 0.5)

multiply_image(image, 0.25, 1, 1)
multiply_image(image, 1, 0.25, 1)
multiply_image(image, 1, 1, 0.25)
multiply_image(image, 0.25, 0.25, 0.25)

multiply_image(image, 1.25, 1, 1)
multiply_image(image, 1, 1.25, 1)
multiply_image(image, 1, 1, 1.25)
multiply_image(image, 1.25, 1.25, 1.25)

multiply_image(image, 1.5, 1, 1)
multiply_image(image, 1, 1.5, 1)
multiply_image(image, 1, 1, 1.5)
multiply_image(image, 1.5, 1.5, 1.5)

gaussian_blur(image, 0.25)
gaussian_blur(image, 0.50)
gaussian_blur(image, 1)
gaussian_blur(image, 2)
gaussian_blur(image, 4)

averageing_blur(image, 5)
averageing_blur(image, 4)
averageing_blur(image, 6)

median_blur(image, 3)
median_blur(image, 5)
median_blur(image, 7)

bilateral_blur(image, 9, 75, 75)
bilateral_blur(image, 12, 100, 100)
bilateral_blur(image, 25, 100, 100)
bilateral_blur(image, 40, 75, 75)

erosion_image(image, 1)
erosion_image(image, 3)
erosion_image(image, 6)

dilatation_image(image, 1)
dilatation_image(image, 3)
dilatation_image(image, 5)

opening_image(image, 1)
opening_image(image, 3)
opening_image(image, 5)

morphological_gradient_image(image, 5)
morphological_gradient_image(image, 10)
morphological_gradient_image(image, 15)

top_hat_image(image, 200)
top_hat_image(image, 300)
top_hat_image(image, 500)

black_hat_image(image, 200)
black_hat_image(image, 300)
black_hat_image(image, 500)

end = datetime.now()

print('Final del proceso')
print('Duracion: {}'.format(end-start))
