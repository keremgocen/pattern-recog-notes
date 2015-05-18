#!/usr/local/bin/python
# coding=utf-8

# k-means clustering method to cluster image pixels into k partitions.
# Method takes input RGB  image and k value, returns a binary image which has
# different labels for different segments. In the output image, each
# input pixel value is replaced with the mean value it is assigned to.

import os
from random import randint
import time

from skimage import io
import numpy as np
import matplotlib.pyplot as plt


# creates RGB pixel object from image pixels
class RGB:
    def __init__(self, rgb_pixel):
        self.r = int(rgb_pixel[0])
        self.g = int(rgb_pixel[1])
        self.b = int(rgb_pixel[2])

    def __str__(self):
        return ''.join("r:" + str(self.r) + " g:" + str(self.g) + " b:" + str(self.b))


# creates Centroid object from coordinates and rgb pixel
class Centroid:
    x = 0
    y = 0
    rgb = RGB(np.zeros(3))

    def __init__(self, xy):
        self.x = xy[0]
        self.y = xy[1]

    def __str__(self):
        return ''.join("x:" + str(self.x) + " y:" + str(self.y) + " rgb:" + str(self.rgb))


# calculate_rgb_distance(pixel, centroid)
# input     - image and centroid Pixel(r,g,b)
# returns   - euclidean distance between two rgb pixels
def calculate_rgb_distance(pixel, center):
    p1 = np.array((pixel.r, pixel.g, pixel.b))
    p2 = np.array((center.r, center.g, center.b))
    return np.linalg.norm(p1 - p2)


# pick_random_xy(shape)
# input     - RGB image matrix shape (rows, columns, rgb)
# returns   - random pixel(x,y) coordinates
def pick_random_xy(shape):
    return [randint(0, shape[0] - 1), randint(0, shape[1] - 1)]


# k_means(image, k)
# input     - rgb image matrix
#           - k number of clusters
#           - difference threshold to stop the algorithm loop(optional)
# returns   - labelImage, labeled image matrix
#           - centroids, k centroids after calculation
def k_means(image, cluster_count, mean_diff_threshold=5.0):
    time_start = time.time()

    # create a matrix of zeros with the same shape to store centroid index assignments for each pixel
    label_image = np.zeros((image.shape[0], image.shape[1]))
    print "label count k:", k
    print "total mean diff threshold:", mean_diff_threshold

    # k-means algorithm
    # STEPS:
    # 0 - Choose k random centroid coordinates
    # For each pixel x:
    # 1 - Calculate rgb distance of x from each centroid
    # 2 - Assign x to closest centroid's cluster
    # For each centroid c:
    # 3 - Calculate rgb means of assigned pixels for that c (move the centroid)
    # repeat until all x are stable

    # 0 - Choose k random centroid coordinates
    centroids = []
    for i in range(cluster_count):
        c = Centroid(pick_random_xy(image.shape))   # set random coordinates
        c.rgb = RGB(image[c.x][c.y])                # set rgb values
        centroids.append(c)

    iteration_count = 0

    while iteration_count < 10:         # repeat until all x are stable, max 10 iterations

        percent_diff = 0.0              # holds percentage in mean change

        # k x 3 dimensional array to hold cluster values, will be used to calculate new means
        cluster_values = np.ones((len(centroids), 3))
        cluster_pixel_counts = np.ones((len(centroids)))

        # 1 - Calculate rgb distance of x from each centroid
        for x in xrange(image.shape[0]):
            for y in xrange(image.shape[1]):

                image_pixel = RGB(image[x][y])
                min_dist = 255
                min_dist_index = 0

                for idx, m in enumerate(centroids):

                    distance = calculate_rgb_distance(image_pixel, m.rgb)

                    if distance < min_dist:
                        min_dist = distance
                        min_dist_index = idx

                # 2 - Assign pixel to closest centroid's cluster(index)
                label_image[x][y] = min_dist_index

                # record pixel values to calculate means
                cluster_values[min_dist_index][0] += image_pixel.r
                cluster_values[min_dist_index][1] += image_pixel.g
                cluster_values[min_dist_index][2] += image_pixel.b
                cluster_pixel_counts[min_dist_index] += 1

        # 3 - Calculate rgb means of assigned pixels for their centroids
        for idx, m in enumerate(centroids):
            
            previous_sum = m.rgb.r + m.rgb.g + m.rgb.b
            
            m.rgb.r = int(cluster_values[idx, 0] / cluster_pixel_counts[idx])
            m.rgb.g = int(cluster_values[idx, 1] / cluster_pixel_counts[idx])
            m.rgb.b = int(cluster_values[idx, 2] / cluster_pixel_counts[idx])
            
            current_sum = m.rgb.r + m.rgb.g + m.rgb.b
            
            # total percent change in cluster means
            if previous_sum > 0:
                percent_diff += (abs((current_sum - previous_sum) / float(previous_sum))) * 100.0
            else:
                percent_diff = 0.0

        iteration_count += 1
        print "total percent difference in means", percent_diff, "iteration", iteration_count

        if mean_diff_threshold > percent_diff > 0.0:
            print "\nStopped at percent difference in means:", percent_diff
            print "total iterations:", iteration_count
            print "total time spent:", '%0.3f ms' % ((time.time() - time_start) * 1000.0)
            print "\ncalculated cluster means:"
            for idx, m in enumerate(centroids):
                print centroids[idx].rgb
            break

    return label_image, centroids


# load RGB image, alpha channel removed
fileName = os.path.join('strawberry.png')
inputImage = io.imread(fileName)

# number of clusters/centroids
k = 4

# run k-means algorithm on input
binaryImage, centroid_array = k_means(inputImage, k)

outputImage = np.zeros_like(inputImage)

# apply cluster centroid means to assigned pixels
for x1 in xrange(binaryImage.shape[0]):
    for y1 in xrange(binaryImage.shape[1]):
        
        outputImage[x1][y1][0] = centroid_array[int(binaryImage[x1][y1])].rgb.r
        outputImage[x1][y1][1] = centroid_array[int(binaryImage[x1][y1])].rgb.g
        outputImage[x1][y1][2] = centroid_array[int(binaryImage[x1][y1])].rgb.b


# create plots and save output image
fig = plt.figure()
imgOrig = plt.subplot2grid((2, 2), (0, 0))
imgOrig.imshow(inputImage, interpolation='nearest')
imgOrig.set_title('Original Image')

imgLabeled = plt.subplot2grid((2, 2), (0, 1))
imgLabeled.imshow(binaryImage, cmap=plt.cm.gray, interpolation='nearest')
imgLabeled.set_title('Labeled Image')

imgOutput = plt.subplot2grid((2, 2), (1, 0), colspan=2)
imgOutput.imshow(outputImage, interpolation='nearest')
imgOutput.set_title('Output Image')

plt.savefig('label-' + str(k) + '-output.png')
plt.show()
