import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from skimage.util import random_noise


class Filters:
    def __init__(self, image_name, box_size=3, kernel_size=3):
        self.image = self.read_image(image_name)
        self.noisy_image = None
        self.filtered_image = None

        self.box_size = box_size
        self.kernel_size = kernel_size

    def read_image(self, image_name):
        return ski.io.imread(image_name)

    def noisy_filter(self, mode='s&p', intensity=0.05):
        noise = None

        if mode == 's&p':
            noise = random_noise(self.image, mode='s&p', amount=intensity)
        else:
            noise = random_noise(self.image, mode='gaussian', var=intensity)

        self.noisy_image = noise

    def box_filter(self):
        H, W, channels = self.image.shape
        box_half = self.box_size // 2

        new_img = np.zeros_like(self.image)
        for c in range(channels):
            for i in range(box_half, H - box_half):
                for j in range(box_half, W - box_half):
                    new_img[i, j, c] = np.mean(
                        self.image[i - box_half:i + box_half + 1, j - box_half:j + box_half + 1, c])

        return new_img

    def median_filter(self):
        H, W, channels = self.image.shape
        kernel_half = self.kernel_size // 2

        new_img = np.zeros_like(self.image)

        for c in range(channels):
            for i in range(kernel_half, H - kernel_half):
                for j in range(kernel_half, W - kernel_half):
                    new_img[i, j, c] = np.median(
                        self.image[i - kernel_half:i + kernel_half + 1, j - kernel_half:j + kernel_half + 1, c])

        return new_img

    def weighted_median(self):
        H, W, channels = self.image.shape
        kernel_half = self.kernel_size // 2

        weights = np.ones((self.kernel_size, self.kernel_size))
        center_weight = self.kernel_size ** 2
        weights[kernel_half, kernel_half] = center_weight

        new_img = np.zeros_like(self.image)

        for c in range(channels):
            for i in range(kernel_half, H - kernel_half):
                for j in range(kernel_half, W - kernel_half):
                    window = self.image[i - kernel_half:i + kernel_half + 1, j - kernel_half:j + kernel_half + 1, c]
                    window = window.flatten() * weights.flatten()
                    new_img[i, j, c] = np.median(np.sort(window))

        return new_img

    def build_plot(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(self.image)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(self.noisy_image)
        axs[1].set_title('Noised Image')
        axs[1].axis('off')

        axs[2].imshow(self.filtered_image)
        axs[2].set_title('Filtered Image')
        axs[2].axis('off')

        plt.show()


if __name__ == "__main__":
    image_paths = ['NeedForSpeed_1.jpg', 'NeedForSpeed_2.jpg', 'NeedForSpeed_3.jpg']

    for path in image_paths:
        image = Filters(path)
        image.noisy_filter(mode='s&p', intensity=0.1)

        # Use different noise removal methods as needed
        image.filtered_image = image.box_filter()
        image.build_plot()
        image.filtered_image = image.median_filter()
        image.build_plot()
        image.filtered_image = image.weighted_median()
        image.build_plot()
