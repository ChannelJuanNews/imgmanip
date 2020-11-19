from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import cv2
import utils
import matplotlib.pyplot as plt
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-c", "--clusters", required=True, type=int,
                help="# of clusters")
args = vars(ap.parse_args())

# load the image and grab its width and height
image = cv2.imread(args["image"])


# convert the image to the correct space
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

# blr image first 
image = cv2.GaussianBlur(image, (3, 3), 0)


# get image dimensions for algorithms
(h, w) = image.shape[:2]

# reshape the image into a feature vector so that k-means can be applied
image = image.reshape((image.shape[0] * image.shape[1], 3))
# apply k-means using the specified number of clusters and
# then create the quantized image based on the predictions
clt = MiniBatchKMeans(n_clusters=args["clusters"])

# grab the labels that were generated with kmeans
# usually will be a list of numbers from 0 to N where N is the number of clusters
labels = clt.fit_predict(image)

# generate a list of unique labels from the labels object
# example list of unique labels looks like this
# [0 1 2 3 4 ]
numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)


# as you can see, the coerced values look similiar to the cluster center values
quantized = clt.cluster_centers_.astype("uint8")[labels]


(hist, hist_labels) = np.histogram(clt.labels_, bins=numLabels)

# normalize the histogram, such that it sums to one
hist = hist.astype("float")
hist /= hist.sum()

# initialize the bar chart representing the relative frequency
# of each of the colors
bar = np.zeros((50, 300, 3), dtype="uint8")

mapped_colors = []
bgr_colors = []
for (percent, color) in zip(hist, clt.cluster_centers_):
    print(percent, color)
    mapped_colors.append((color.astype("uint8").tolist(), percent))
    bgr_colors.append(color.astype("uint8").tolist())

#hex_colors = utils.get_hex_colors(bgr_colors)

# reshape the feature vectors to images for opencv to use
quantized = quantized.reshape((h, w, 3))
image = image.reshape((h, w, 3))

masks = []
for i in range(args["clusters"]):
    # define the mask according to the stored colors
    mask = cv2.inRange(quantized, np.array(bgr_colors[i]), np.array(bgr_colors[i]))

    #cv2.namedWindow('MASK IS', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('MASK IS', 1000,1000)
    #cv2.imshow("MASK IS", mask)


    # get
    res = cv2.bitwise_and(quantized, quantized, mask=mask)

    # create white background for later use  
    white = np.full(res.shape, 255, dtype=np.uint8)


    # turn the resulting mask to gray color space
    gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # threshold the values
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    dilation = cv2.dilate(thresh, kernel, iterations=1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)

    dilation = cv2.dilate(gradient, kernel, iterations=1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)

    (contours, h) = cv2.findContours(
        gradient.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        # make sure it is a closed loop
        if cv2.contourArea(contour) > cv2.arcLength(contour, True):
            if cv2.contourArea(contour) >= 500:
                M = cv2.moments(contour)
                cX = int(M["m10"]/M["m00"])
                cY = int(M["m01"]/M["m00"])

                # create a white background image 
                cv2.drawContours(white, contour, -1, (0, 0, 0), 1)
                #cv2.putText(white, "{}".format(i+1), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                

    #cv2.imwrite('mask-{}.png'.format('-'.join(str(s) for s in bgr_colors[i])), res)
    #cv2.imshow("test", white)
    masks.append(white.copy())
    cv2.imwrite('mask-{}.png'.format(i), white)
    #cv2.imshow("thresholded image", np.hstack([thresh, dilation, closing,  gradient]))
    #cv2.imshow("final_result", white)
    #cv2.waitKey()




final_image = masks[0]



for i in range(args["clusters"]):
    final_image = cv2.bitwise_and(final_image, masks[i])


#change the kernel 
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(final_image,kernel,iterations = 1)

out = cv2.bitwise_not(erosion)
#out = cv2.morphologyEx(out, cvgit 2.MORPH_CLOSE, kernel)
#out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
#out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
#out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
#out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)



out = cv2.Canny(negative, 250, 255)
out = cv2.bitwise_not(out)




cv2.namedWindow('final image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('final image', 2000,2000)
cv2.imshow("final image", out)
cv2.waitKey()
exit()
    

  

exit()

# python3 quantized.py -i andrea-starter-red.jpg -c 5
