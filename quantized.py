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

# convert the image from the RGB color space to the L*a*b*
# color space -- since we will be clustering using k-means
# which is based on the euclidean distance, we'll use the
# L(ightness) * A (color from Green to magenta * B (color from blue to yellow)
# This color space where the euclidean distance implies perceptual meaning
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

# do canny edge detection to compare after post processing
canny_first = cv2.Canny(image, 50, 50)
white_background = np.full(image.shape, 255, dtype=np.uint8) 
mask = canny_first != 255
canny_first = white_background * (mask[:,:,None].astype(white_background.dtype))




# get image dimensions for algorithms 
(h, w) = image.shape[:2]


# reshape the image into a feature vector so that k-means can be applied
image = image.reshape((image.shape[0] * image.shape[1], 3))
# apply k-means using the specified number of clusters and
# then create the quantized image based on the predictions
clt = MiniBatchKMeans(n_clusters = args["clusters"])

# grab the labels that were generated with kmeans 
# usually will be a list of numbers from 0 to N where N is the number of clusters
labels = clt.fit_predict(image)

# generate a list of unique labels from the labels object
# example list of unique labels looks like this 
# [0 1 2 3 4 ] 
numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)


# 2 step process in a one liner 
# 1. 
# the cluster_center should be an array of length N where N is the number of clusters
# and each item in the array/list is the quantized colorspace value (sort of)
# as each (R,G,B) OR (L,A,B) value will be a float and not an intger
# we will need to flatten/round the values later
# example cluster centers looks like this 
#  [
#   [ 49.94716975 129.40613757 133.99778298]
#   [128.31349142 123.07266482 144.54975432]
#   [174.47337638 133.6048346  132.49002635]
#   [137.25054687 190.08033083 190.13393293]
#   [ 89.667345   123.04586927 137.8495374 ]
# ]
# 2. coerces the pixels that are labeled as part of a
# cluster into the assoicated color value. The output is an array of 
# length N where N is the total number of pixels
# example coercing looks like this 
# [ 
#  [ 88 123 137]
#  [ 88 123 137]
#  [ 88 123 137]
#  ...
#  [ 47 129 134]
#  [ 47 129 134]
#  [ 47 129 134]
# ]
# as you can see, the coerced values look similiar to the cluster center values
quantized = clt.cluster_centers_.astype("uint8")[labels]


# map all the labels into their respective bins 
# this returns a tuple with 2 values 
# first is a list with the number of pixels labeled to each bin 
# second is a list of the labels 
# example histogram looks like this 
# ([5755879 8343635 1380439 2686080 3952367] [0 1 2 3 4 5]
(hist, hist_labels) = np.histogram(clt.labels_, bins = numLabels)

# normalize the histogram, such that it sums to one
hist = hist.astype("float")
hist /= hist.sum()

# initialize the bar chart representing the relative frequency
# of each of the colors
bar = np.zeros((50, 300, 3), dtype = "uint8")

# create empty mapped colors array to contain the color space values for each 
# cluster that was generated with kmeans. This will save the colorspace value 
# as well as the frequency that that color showed up as a percentage of 100
mapped_colors = []
bgr_colors = []
for (percent, color) in zip(hist, clt.cluster_centers_):
    print(percent, color)
    mapped_colors.append( (color.astype("uint8").tolist(), percent ) )
    bgr_colors.append( color.astype("uint8").tolist() )
   


#hex_colors = utils.get_hex_colors(bgr_colors)


# reshape the feature vectors to images for opencv to use 
quantized = quantized.reshape((h, w, 3))
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
    # define the mask according to the stored colors 
    mask = cv2.inRange(quantized, np.array(bgr_colors[i]), np.array(bgr_colors[i]))
   
    
    
    #cv2.namedWindow('MASK IS', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('MASK IS', 1000,1000)
    #cv2.imshow("MASK IS", mask)
    
    # get 
    res = cv2.bitwise_and(quantized, quantized, mask = mask)
 


    # turn the resulting mask to gray color space 
    gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # threshold the values 
    ret, thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    
    # get the blue, green and red channels from the starter result
    b, g, r = cv2.split(res)

    # some more rgba stuff
    rgba = [b,g,r, thresh]

    # this is the rgba image (has alpha channel)
    RGBA_RES = cv2.merge(rgba,4)
    cv2.imwrite("test-{}.png".format(i), RGBA_RES)

    
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        #center_x = int( (w-x)/2 )
        #center_y = int( (h-y)/2 )
        print(x, y, w, h)
        #cv2.putText(res, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (36,255,12), 2)

    cv2.drawContours(res, contours, -1, (105, 105, 105))

    #cv2.imshow("bb img", res)
    #cv2.waitKey()
 


    cv2.imwrite('mask-{}.png'.format('-'.join(str(s) for s in bgr_colors[i])), RGBA_RES)
    masks.append( res.copy())
    #cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('result', 1000,1000)
    #cv2.imshow("result", res)
    #cv2.waitKey()
    
final_result = quantized.copy()


#final_result = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

for i in range(args["clusters"]):
    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = masks[i].shape
    roi = final_result[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(masks[i] , cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY)


    


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

#quant = cv2.cvtColor(quant, cv2.COLOR_BGR2RGB)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


cv2.namedWindow('canny', cv2.WINDOW_NORMAL)
cv2.resizeWindow('canny', 2000,2000)
canny = cv2.Canny(final_result, 100, 250)
white_background = np.full(final_result.shape, 255, dtype=np.uint8) 
mask = canny != 255
final_canny = white_background * (mask[:,:,None].astype(white_background.dtype))

cv2.imshow("final canny", canny_first)
cv2.waitKey()


canny_diff = cv2.bitwise_and( final_result,canny_first, mask = None) 


#cv2.imshow('canny', final_canny)
#cv2.waitKey()

#cv2.namedWindow('images', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('images', 2000,2000)
#cv2.imshow("images", np.hstack([image, quant, final_result]))
cv2.imwrite('output.jpg', np.hstack([image, quantized, final_result, final_canny, canny_first, canny_diff ]) )
#cv2.waitKey()



#python3 quantized.py -i andrea-starter-red.jpg -c 5