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
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
(h, w) = image.shape[:2]
# convert the image from the RGB color space to the L*a*b*
# color space -- since we will be clustering using k-means
# which is based on the euclidean distance, we'll use the
# L*a*b* color space where the euclidean distance implies
# perceptual meaning

# reshape the image into a feature vector so that k-means
# can be applied
image = image.reshape((image.shape[0] * image.shape[1], 3))
# apply k-means using the specified number of clusters and
# then create the quantized image based on the predictions
clt = MiniBatchKMeans(n_clusters = args["clusters"])

labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]

hist = utils.centroid_histogram(clt)

rgb_colors = utils.get_colors(hist, clt.cluster_centers_)

hex_colors = utils.get_hex_colors(rgb_colors)


# reshape the feature vectors to images
quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))


# convert from L*a*b* to RGB
#quant = cv2.cvtColor(quant, cv2.COLOR_BGR2RGB)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#print(quant)

# display the images and wait for a keypress

#plt.figure()

#data = [100, 100, 100, 100, 100]
#barchart = plt.bar(hex_colors, data)
#barchart[0].set_color(hex_colors[0])
#barchart[1].set_color(hex_colors[1])
#barchart[2].set_color(hex_colors[2])
#barchart[3].set_color(hex_colors[3])
#barchart[4].set_color(hex_colors[4])
#plt.show()


masks = []

# use the rgb colors to mask the quantized image 
for i in range(args["clusters"]):
    mask = cv2.inRange(quant, np.array(rgb_colors[i]), np.array(rgb_colors[i]))
    #cv2.namedWindow('MASK IS', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('MASK IS', 1000,1000)
    #cv2.imshow("MASK IS", mask)
    
    res = cv2.bitwise_and(quant, quant, mask = mask)
    # turn the resulting mask to gray color space 
    gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res, contours, -1, (105, 105, 105))
    
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    cv2.imwrite('mask-{}.jpg'.format(i), res)
    masks.append(res.copy())
    #cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('result', 1000,1000)
    #cv2.imshow("result", res)
    #cv2.waitKey()
    


final_result = quant.copy()
final_result = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

for i in range(args["clusters"]):
    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = masks[i].shape
    roi = final_result[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(masks[i] , cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 50, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)


    # Now black-out the area of logo in ROI
    bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    fg = cv2.bitwise_and(masks[i],masks[i],mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(bg, fg)
    final_result[0:rows, 0:cols ] = dst
    #cv2.namedWindow('FINAL RESULT', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('FINAL RESULT', 1000,1000)
    #cv2.imshow('FINAL RESULT', final_result)
    #cv2.waitKey()

quant = cv2.cvtColor(quant, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#cv2.namedWindow('images', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('images', 2000,2000)
#cv2.imshow("images", np.hstack([image, quant, final_result]))
cv2.imwrite('output.jpg', np.hstack([image, quant, final_result]) )
#cv2.waitKey()



#python3 quantized.py -i andrea-starter-red.jpg -c 5