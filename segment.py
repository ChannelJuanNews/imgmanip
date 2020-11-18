import cv2 

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from sklearn.cluster import KMeans
import argparse

import cv2
import utils

# flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.reshape((image.shape[0] * image.shape[1], 3))
clt = KMeans(n_clusters = aaurgs["clusters"])
clt.fit(image)







hist = utils.centroid_histogram(clt)
colors = utils.get_colors(hist, clt.cluster_centers_)




print(colors)

# python3 segment.py -i img.jpg -c 5