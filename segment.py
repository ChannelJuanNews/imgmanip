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


img = cv2.imread('./img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.reshape((image.shape[0] * image.shape[1], 3))
clt = KMeans(n_clusters = args["clusters"])
clt.fit(image)

print(clt.labels_)

hist = utils.centroid_histogram(clt)
bar = utils.plot_colors(hist, clt.cluster_centers_)
# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()

# python3 segment.py -i img.jpg -c 5