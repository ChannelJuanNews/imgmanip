import numpy as np
import cv2
def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
	# return the histogram
	return hist

def get_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
	# loop over the percentage of each cluster and the color of
	# each cluster
	mapped_colors = []
	for (percent, color) in zip(hist, centroids):
		mapped_colors.append(color.astype("uint8").tolist())
		#print(color.astype("uint8").tolist())
		# plot the relative percentage of each cluster
		#endX = startX + (percent * 300)
		
		#cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
		#startX = endX
	
	# return the bar chart
	return mapped_colors

def get_rgb_colors(colors):
	hex_colors = []
	for color in colors:
		hex =  '#%02x%02x%02x' % tuple(color)
		hex_colors.append(hex) 
	return hex_colors 
	