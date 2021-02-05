import io

import PIL
import matplotlib.image as img
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
import numpy as np
import sys


def read_image_as_jpg(file):
    image_read = PIL.Image.open(file)
    image_format = image_read.format

    if image_format == 'JPEG':
        image = img.imread(file, format='jpg')

    else:
        io_bytes = io.BytesIO()
        image_converted = image_read.convert('RGB')
        image_converted.save(io_bytes, format='JPEG')
        image = img.imread(io_bytes, format='jpg')

    return image


def normalize_image(image):
    red_values = image[:, :, 0]
    green_values = image[:, :, 1]
    blue_values = image[:, :, 2]

    sum_of_rgb_values = red_values.astype(int) + green_values.astype(int) + blue_values.astype(int)
    sum_of_rgb_values[sum_of_rgb_values == 0] = 1

    normalized_image = image.copy()

    normalized_image[:, :, 0] = red_values / sum_of_rgb_values * 255.0
    normalized_image[:, :, 1] = green_values / sum_of_rgb_values * 255.0
    normalized_image[:, :, 2] = blue_values / sum_of_rgb_values * 255.0

    return normalized_image


def cluster(image):
    pixels = image.reshape((image.shape[0] * image.shape[1], 3))

    initial_cluster_centers = np.array([[67, 90, 94], [87, 93, 73], [48, 89, 144], [76, 100, 76], [78, 89, 85]])
    k_means_result = KMeans(n_clusters=5, n_init=1, init=initial_cluster_centers).fit(pixels)
    labels = k_means_result.labels_

    return labels


def generate_clustered_image(labels, image):
    labels = labels.reshape(image.shape[0], image.shape[1])
    result = image.copy()

    color_red = [255, 0, 0]
    color_green = [0, 255, 0]
    color_blue = [0, 0, 255]
    color_dark_green = [0, 100, 0]

    result[np.where(labels == 2)] = color_blue
    result[np.where(labels == 4)] = color_red
    result[np.where(labels == 0)] = color_dark_green
    result[np.where(labels == 1)] = color_green
    result[np.where(labels == 3)] = color_green

    return result


def show_image(result):
    plt.imshow(result)
    plt.axis("off")
    plt.show()


def save_image(result, name):
    save_under_path = name + "_clustered.png"
    plt.imsave(save_under_path, result)
    print("Result saved as " + save_under_path)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path_to_image = sys.argv[1]
    else:
        path_to_image = input("Please enter the image path: ")

    image_jpg = read_image_as_jpg(path_to_image)
    image_normalized = normalize_image(image_jpg)

    labels = cluster(image_normalized)
    result_image = generate_clustered_image(labels, image_jpg)

    show_image(result_image)
    save_image(result_image, path_to_image)

