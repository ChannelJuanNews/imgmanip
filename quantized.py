from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import cv2
import utils
import matplotlib.pyplot as plt
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())

# load the image and grab its width and height
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
# convert the image from the RGB color space to the L*a*b*
# color space -- since we will be clustering using k-means
# which is based on the euclidean distance, we'll use the
# L*a*b* color space where the euclidean distance implies
# perceptual meaning
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# reshape the image into a feature vector so that k-means
# can be applied
image = image.reshape((image.shape[0] * image.shape[1], 3))
# apply k-means using the specified number of clusters and
# then create the quantized image based on the predictions
clt = MiniBatchKMeans(n_clusters = args["clusters"])

labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]

hist = utils.centroid_histogram(clt)
colors = utils.get_colors(hist, clt.cluster_centers_)
print(colors)
colors = utils.get_rgb_colors(colors)
print(colors)

# reshape the feature vectors to images
quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))


# convert from L*a*b* to RGB
quant = cv2.cvtColor(quant, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# display the images and wait for a keypress
cv2.imshow("image", np.hstack([image, quant]))
plt.figure()
data = [100, 100, 100, 100, 100]
barchart = plt.bar(colors, data)
barchart[0].set_color(colors[0])
barchart[1].set_color(colors[1])
barchart[2].set_color(colors[2])
barchart[3].set_color(colors[3])
barchart[4].set_color(colors[4])

plt.show()

cv2.waitKey(0)