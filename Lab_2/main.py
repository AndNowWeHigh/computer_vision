import cv2
import numpy as np
from numba import njit

image = cv2.imread('image.jpg')

#Функція, реалізує ефект ЧБ
Array_grey = []

for Color_Pixel in image:
    list_1 = []
    for grey_pixel in Color_Pixel:
        list_1.append(sum(grey_pixel) // 3)
    Array_grey.append(list_1)

# отримуємо маємо масив із відтінками сірого
grey_img = np.array(Array_grey, dtype=np.uint8)
print(len(grey_img.shape))
print(len(image.shape))

#Функція, реалізує ефект згортки

@njit
def the_convolution_filter(image, kernel):
    image_height, image_width, num_channels = image.shape
    kernel_height, kernel_width = kernel.shape

    # Визначаємо розмір вихідного зображення
    output = np.zeros((image_height - kernel_height + 1, image_width - kernel_width + 1, num_channels), dtype=np.uint8)

    # Здійснюємо згортку
    for h in range(output.shape[0]):
        for w in range(output.shape[1]):
            for c in range(output.shape[2]):
                output[h, w, c] = np.sum(image[h:h + kernel_height, w:w + kernel_width, c] * kernel)

    return output

@njit
def the_convolution_filter_grey(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Визначаємо розмір вихідного зображення
    output = np.zeros((image_height - kernel_height + 1, image_width - kernel_width + 1), dtype=np.uint8)

    # Здійснюємо згортку
    for h in range(output.shape[0]):
        for w in range(output.shape[1]):
            output[h, w] = np.sum(image[h:h + kernel_height, w:w + kernel_width] * kernel)

    return output


# 1 Зсув зображення на 10 пiкселiв вправо та 20 вниз.

def shift_image(shift_x, shift_y):
    # Застосовуємо зсув для кожного каналу кольору
    shifted_image1 = np.zeros_like(image)

    for i in range(image.shape[2]):
        shifted_image1[:, :, i] = np.roll(image[:, :, i], (shift_y, shift_x))

    return shifted_image1

shift_image2 = shift_image(10, 20)
cv2.imwrite('shift.jpg', shift_image2)
cv2.waitKey(0)


# 2 Iнверсiя. ядро (фільтр)
kernel = np.array([[0, 0, 0],
                   [0, -1, 0],
                   [0, 0, 0]])


filtered_image = the_convolution_filter(image, kernel)

# Виводимо вхідне та вихідне зображення
cv2.imwrite('Inversion.jpg', filtered_image)

cv2.waitKey(0)

# 3 Згладжування по Гауссу 11х11

def gaussian_kernel(size, sigma=1):
    kernel = np.fromfunction(lambda x, y: (1/ (2*np.pi*sigma**2)) * np.exp(- ((x-size//2)**2 + (y-size//2)**2) / (2*sigma**2)),(size, size))
    return kernel / np.sum(kernel)

# Розмір матриці Гауса
size = 11

# Створюємо матрицю Гауса
kernel_3 = gaussian_kernel(size, sigma=1)

filtered_image = the_convolution_filter(image, kernel_3)

# Виводимо вхідне та вихідне зображення
cv2.imwrite('Gauss.jpg', filtered_image)

cv2.waitKey(0)


# 4 Розмиття "рух по дiагоналi" менше 7х7
def kernel_diagonal_blur(x, y):
    kernel = np.zeros((x, y), dtype=np.uint8)

    for i in range(x):
        kernel[i][i] = 1
    return kernel / x
diagonal_blur = the_convolution_filter(image, kernel_diagonal_blur(7, 7))

cv2.imwrite('Blur_diagonal_movement.jpg', diagonal_blur)
cv2.waitKey(0)

# 5 Пiдвищення рiзкостi.
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
filtered_image = the_convolution_filter_grey(grey_img, kernel)

cv2.imwrite('sharpness.jpg', filtered_image)
cv2.waitKey(0)

# 6 Фiльтр Собеля
kernel = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
filtered_image = the_convolution_filter_grey(grey_img, kernel)
cv2.imwrite('Sobel_filter.jpg', filtered_image)
cv2.waitKey(0)

# 7 Фiльтр границi
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])
filtered_image = the_convolution_filter_grey(grey_img, kernel)
cv2.imwrite('boundary.jpg', filtered_image)
cv2.waitKey(0)

# 8 Цiкавий фiльтр який ви вигадали самi
kernel = (np.array([[-1, -3, 0],
                   [-3, 0, 3],
                   [0, 3, 1]])) / 9
filtered_image = the_convolution_filter(image, kernel)
cv2.imwrite('MyFilter.jpg', filtered_image)
cv2.waitKey(0)


#
#
#
#

