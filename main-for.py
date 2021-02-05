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


def norm_image(image):
    blue_values = image[:, :, 0]
    green_values = image[:, :, 1]
    red_values = image[:, :, 2]

    sum_of_rgb_values = blue_values.astype(int) + green_values.astype(int) + red_values.astype(int)
    sum_of_rgb_values[sum_of_rgb_values == 0] = 1

    result = image.copy()

    result[:, :, 0] = blue_values / sum_of_rgb_values * 255.0
    result[:, :, 1] = green_values / sum_of_rgb_values * 255.0
    result[:, :, 2] = red_values / sum_of_rgb_values * 255.0

    return result


def cluster(result):
    pixels = result.reshape((result.shape[0] * result.shape[1], 3))

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
    color_light_green = [0, 100, 0]

    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            if (labels[x, y]) == 0:
                result[x, y] = color_light_green
            elif (labels[x, y]) == 1:
                result[x, y] = color_green
            elif (labels[x, y]) == 2:
                result[x, y] = color_blue
            elif (labels[x, y]) == 3:
                result[x, y] = color_green
            elif (labels[x, y]) == 4:
                result[x, y] = color_red

    return result


def show_image(result):
    plt.imshow(result)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    # access command-line-argument
    path_to_image = sys.argv[1]

    start = time.time()

    image_jpg = read_image_as_jpg(path_to_image)
    image_normed = norm_image(image_jpg)
    labels = cluster(image_normed)
    result_image = generate_clustered_image(labels, image_jpg)
    show_image(result_image)

    end = time.time()
    print('{:5.3f}s'.format(end - start))
