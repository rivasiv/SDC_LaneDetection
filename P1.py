#!/usr/bin/python

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import sys

#%matplotlib inline

import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

from scipy import stats
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    x_l, y_l, x_r, y_r = [], [], [] ,[]

    y1 = int(1.1 * YU)
    y2 = H
    if (debug == 1):
        for line in lines:
            for (x1, y1, x2, y2) in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness=2)

    for line in lines:
        slope = slope_line(line)

        # Filter out left lane markers ensuring none of the points cross right top x boundary
        if((slope > SLOPE_THRESHOLD) and (line[0][0] > (W/2 - XU_OFFSET))):
            x_r.append(line[0][0])
            x_r.append(line[0][2])
            y_r.append(line[0][1])
            y_r.append(line[0][3])
        # Filter out right lane markers ensuring none of the points cross left top x boundary
        elif((slope < (-1 * SLOPE_THRESHOLD)) and (line[0][0] < (W/2 + XU_OFFSET))):
            x_l.append(line[0][0])
            x_l.append(line[0][2])
            y_l.append(line[0][1])
            y_l.append(line[0][3])

    # Plot left lane marker
    if((len(x_l) != 0) and (len(y_l) != 0)):
        slope_l, intercept_l, r_value, p_value, std_err = stats.linregress(x_l,y_l)
        x1 = int((y1 - intercept_l)/slope_l)
        x2 = int((y2 - intercept_l)/slope_l)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness=8)

    # Plot right lane marker
    if((len(x_r) != 0) and (len(y_r) != 0)):
        slope_r, intercept_r, r_value, p_value, std_err = stats.linregress(x_r,y_r)
        x1 = int((y1 - intercept_r)/slope_r)
        x2 = int((y2 - intercept_r)/slope_r)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness=8)

    # Try to detect both lane markers and return 1 if either/both are missing
    if((len(x_l) == 0) or (len(x_r) == 0)):
        if(debug == 1):
            for line in lines:
                for (x1, y1, x2, y2) in line:
                    cv2.line(img, (x1, y1), (x2, y2), color=[255, 255, 255], thickness=2)
                    print(slope_line(line))
                    cv2.imshow("Lines_internal_debug", img)
                    cv2.waitKey(0)
        missing_markers = (len(x_l) == 0) + (len(x_r) == 0)
        print(str(missing_markers) + "lane markers not detected for frame " + str(frame))
        return 1

    #return 0 if both lane markers were detected
    return 0

def slope_line(line):
    dx = line[0][2] - line[0][0]
    dy = line[0][3] - line[0][1]
    return (dy/dx)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, a=0.8, b=1., l=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + lambda
    NOTE: initial_img and img must be the same shape!
    #"""
    return cv2.addWeighted(initial_img, a, img, b, l)

def show_image(name, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow(name , image)

#Edge detection
CANNY_LOW = 50
CANNY_HIGH = 150

# Noise filter
GAUSS_KER = 3

#Hough transform params
HOUGH_THRESHOLD = 10
HOUGH_MIN_LEN = 20
HOUGH_MAX_GAP = 80

#Lane detection slope threshold
SLOPE_THRESHOLD = 0.4

#Image params
H = 0
W = 0
YU = 0
XU_OFFSET = 0
ROI_VERTICES = np.array([[0, H], [0, H], [0, YU], [0, YU]], np.int32)
frame = 0

debug = 0
global_init = 0

def init_globals(image):
    global H, W, YU, XU_OFFSET, global_init, ROI_VERTICES
    H = image.shape[0]
    W = image.shape[1]
    XU_OFFSET = int(W/25)
    YU = int(H/2 + H/12)
    xu1 = int(W/2 + XU_OFFSET)
    xu2 = int(W/2 - XU_OFFSET)
    xl1 = int(W/20)
    xl2 = int(W - W/20)
    ROI_VERTICES = np.array([[xl1, H], [xl2, H], [xu1, YU], [xu2, YU]], np.int32)
    global_init = 1

def process_image(image):
    # track the frame being operated on
    global frame
    frame = frame + 1

    # initialize global variables depending on image params
    if(global_init == 0):
        init_globals(image)

    # create a copy of original image to render final output
    original_image = image.copy()

    # Extract part of image we are interested in to avoid unwanted computations
    image = region_of_interest(image, [ROI_VERTICES])

    # Convert to greyscale to avoid further computations
    image_grey = grayscale(image)

    # HSV representation of yellow is [30, 255, 255]
    # filtering out yellow with some delta colors around it
    hyl = np.array([20, 50, 50], dtype = "uint8")
    hyu = np.array([40, 255, 255], dtype="uint8")

    # Create a mask to extract yello color from frame
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask_y = cv2.inRange(image_hsv, hyl, hyu)

    # Extract white/bright color from greyscale image. This will give data for while lane markers
    mask_w = cv2.inRange(image_grey, 200, 255)

    # Combine the mask for white and yellow
    mask = cv2.bitwise_or(mask_w, mask_y)

    # Extract white and yellow colored pixels from image frame
    image_yw_mask = cv2.bitwise_and(image_grey, mask)

    # From grey image extract edges
    image_canny = canny(image_yw_mask, CANNY_LOW, CANNY_HIGH)

    # Blur out image to reduce effect of noise
    image_gb = gaussian_blur(image_canny, GAUSS_KER)

    # Find lines(lane markers) in the image which fit the edges detected by canny filter
    image_line = hough_lines(image_gb, 1, np.pi/180, HOUGH_THRESHOLD, HOUGH_MIN_LEN, HOUGH_MAX_GAP);

    # overlay the lane markers on original image
    image_weighted = weighted_img(image_line, original_image)

    # render final image
    show_image("Final", image_weighted)

    if(debug == 1):
        show_image("Input", image)
        cv2.imshow("Grey Input", image_grey)
        cv2.imshow("Yellow mask", mask_y)
        cv2.imshow("White mask", mask_w)
        cv2.imshow("Y + W Image", image_yw_mask)
        cv2.imshow("Canny edges", image_canny)
        cv2.imshow("Gaussian Blur", image_gb)
        show_image("Line", image_line)
        cv2.waitKey(0)
    cv2.waitKey(1)
    return image_weighted

white_output = 'white.mp4'
clip1 = VideoFileClip("challenge.mp4")
#clip1 = VideoFileClip("P1_example.mp4")
#clip1 = VideoFileClip("raw-lines-example.mp4")
#clip1 = VideoFileClip("solidWhiteRight.mp4")
#clip1 = VideoFileClip("solidYellowLeft.mp4")
#clip1 = VideoFileClip("white.mp4")

white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

'''
import os
files = os.listdir("test_images/")

for filename in files:
    #reading in an image
    image = mpimg.imread('test_images/' + filename)
    print('This image is:', type(image), 'with dimesions:', image.shape)
    output_image = process_image(image)
    mpimg.imsave('test_images/lanes_'+filename, output_image)
'''
